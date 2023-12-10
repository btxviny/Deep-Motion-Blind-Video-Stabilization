import numpy as np
import cv2
import argparse
import os 
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms ,models

from models.models import ENet,UNet,TempDiscriminator3D
from data.datagen_stage3 import DataGenerator
from config import SKIP

 
def parse_args():
    parser = argparse.ArgumentParser(description='DMBVS-stage1-train')
    parser.add_argument('--model', type=str, help='Model Architecture')
    parser.add_argument('--ckpt_dir', type=str, help='path to stage 1 weights directory')
    parser.add_argument('--shape', nargs='+', type=int, help='H W C values', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--txt_path', type=str, help='path to training list')
    return parser.parse_args()

def load_checkpoint(generator,discriminator,g_optimizer,d_optimizer, ckpt_dir):
    #check if directory exists, if not load weights from previous session
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        os.makedirs(os.path.join(ckpt_dir,'generator/'))
        os.makedirs(os.path.join(ckpt_dir,'discriminator/'))
        prev_ckpt_dir = './stage2/generator/'
        ckpts = [x for x in os.listdir(prev_ckpt_dir) if x.endswith('.pth')]
        ckpts = sorted(ckpts, key = lambda x : x.split('.')[0].split('_')[1]) #sort
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(prev_ckpt_dir,latest))
        generator.load_state_dict(state_dict['model'])
        print(f'Loaded weights:{latest} from previous stage.\nCheckpoint directory {ckpt_dir} created.')
    else:
        #load generator
        gen_path = os.path.join(ckpt_dir,'generator/')
        ckpts = [x for x in os.listdir(gen_path) if x.endswith('.pth')]
        ckpts = sorted(ckpts, key=lambda x: int(x.split('.')[0].split('_')[1]))  # sort by epoch number
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(gen_path, latest))
        generator.load_state_dict(state_dict['model'])
        starting_epoch = state_dict['epoch'] + 1
        g_optimizer.load_state_dict(state_dict['optimizer'])
        print(f'Loaded generator: {latest}')
        #load critic
        disc_path = os.path.join(ckpt_dir,'critic/')
        ckpts = [x for x in os.listdir(disc_path) if x.endswith('.pth')]
        ckpts = sorted(ckpts, key=lambda x: int(x.split('.')[0].split('_')[1]))  # sort by epoch number
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(disc_path, latest))
        discriminator.load_state_dict(state_dict['model'])
        d_optimizer.load_state_dict(state_dict['optimizer'])
        print(f'Loaded discriminator: {latest}')

def get_data_loader(shape, txt_path, skip, batch_size):
    class IterDataset(data.IterableDataset):
        def __init__(self, data_generator):
            super(IterDataset, self).__init__()
            self.data_generator = data_generator
        def __iter__(self):
            return iter(self.data_generator())
    data_gen = DataGenerator(shape= shape, txt_path= txt_path, skip=skip)
    train_ds = iter(data_gen())
    return train_ds

#loss functions
def contextual_loss(x, y, h=0.5):
    """Computes contextual loss between x and y.
    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).
    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    assert x.size() == y.size()
    N, C, H, W = x.size()   # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).
    y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)
    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)           # (N, H*W, H*W)
    d = 1 - cosine_sim                                  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data 
    d_min, _ = torch.min(d, dim=2, keepdim=True)        # (N, H*W, 1)
    # Eq (2)
    d_tilde = d / (d_min + 1e-5)
    # Eq(3)
    w = torch.exp((1 - d_tilde) / h)
    # Eq(4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)       # (N, H*W, H*W)
    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))
    return cx_loss
def perceptual_loss(x, y):
    x_norm = x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    y_norm = y / torch.sqrt(torch.sum(y**2, dim=1, keepdim=True))
    return torch.sqrt(torch.sum((x_norm - y_norm)**2)) ** 2 / x.numel()
def contrastive_motion_loss(stable, unstable, generated):
    batch_size = stable.shape[0]
    stable = preprocess(stable)
    unstable = preprocess(unstable)
    generated = preprocess(generated)
    A = encoder(stable).view(batch_size,-1)
    A = A / torch.sqrt(torch.sum(A**2, dim=1, keepdim=True))
    P = encoder(generated).view(batch_size,-1)
    P = P / torch.sqrt(torch.sum(P**2, dim=1, keepdim=True))
    N = encoder(unstable).view(batch_size,-1)
    N = N / torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))

    d1 = torch.mean(torch.sqrt(torch.sum(torch.pow(A - P,2),dim =1)),dim = 0,keepdim=True).to(device) #euclidean distance of vectors
    d2 = torch.mean(torch.sqrt(torch.sum(torch.pow(A - N,2),dim =1)),dim = 0,keepdim=True).to(device)#euclidean distance of vectors
    return torch.max(d1 - d2 + 1, 0).values

binary_cross_entropy = nn.BCELoss()

def preprocess(tensor, resize_shape=(128, 171), crop_shape=(112, 112),
                        mean=[0.155, 0.161, 0.153], std=[0.228, 0.231, 0.226]):
    """
    Apply transforms to each 3D slice of the 4D tensor along the time dimension.

    Args:
        tensor (torch.Tensor): 4D tensor of shape [B, C, T, H, W].
        resize_shape (tuple): The target size for resizing each 3D slice (H, W).
        crop_shape (tuple): The target size for center cropping each 3D slice (H, W).
        mean (list): List of mean values for normalization.
        std (list): List of standard deviation values for normalization.

    Returns:
        torch.Tensor: Transformed 4D tensor of shape [B, C, T, H, W].
    """
    transforms_3d = transforms.Compose([
        transforms.Resize(resize_shape, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(crop_shape),
        transforms.Normalize(mean=mean, std=std),
    ])

    transformed_slices = []
    for t in range(tensor.size(2)):
        transformed_slice = transforms_3d(tensor[:, :, t])
        transformed_slices.append(transformed_slice.unsqueeze(2))  # Add the time dimension back
    return torch.cat(transformed_slices, dim=2)

def train(generator, g_optimizer, discriminator, d_optimizer,train_ds, writer, starting_epoch, args):
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    EPOCHS = 20
    g_running_loss = 0
    d_running_loss = 0
    dataset_len = 8000
    for epoch in range(starting_epoch, EPOCHS):
        for idx,batch in enumerate(train_ds):
            torch.cuda.empty_cache()
            input_sequence, unstable_sequence, stable_sequence = batch
            generated_sequence = torch.zeros(1,3,8,args.shape[0],args.shape[1]).float()
            g_loss = 0
            for k in range(8):
                x = input_sequence[k].to(device)
                y = stable_sequence[:,:,k,:,:].to(device)
                y_hat = generator(x)
                generated_sequence[:,:,k,:,:] = y_hat.cpu()
                # compute image losses
                #get embeddings
                feat1 = mobile(y_hat)
                feat2 = mobile(y)
                percept_loss = perceptual_loss(feat1,feat2)
                context_loss = contextual_loss(feat1, feat2)
                g_loss += percept_loss + context_loss
        
            # Update temporal discriminator
            d_optimizer.zero_grad()
            fake_prediction = discriminator(generated_sequence.detach().to(device))
            fake_labels = torch.zeros_like(fake_prediction)
            real_prediction = discriminator(stable_sequence.to(device))
            real_labels = torch.ones_like(real_prediction)
            predictions = torch.cat([fake_prediction, real_prediction], dim=0)
            labels = torch.cat([fake_labels, real_labels], dim=0)
            d_loss = binary_cross_entropy(predictions, labels)
            d_loss.backward()
            d_optimizer.step()
            
            #Update Generator
            generated_sequence = generated_sequence.to(device)
            unstable_sequence = unstable_sequence.to(device)
            stable_sequence = stable_sequence.to(device)
            #with torch.no_grad():
            score = discriminator(generated_sequence)
            labels = torch.ones_like(score).to(device)
            generator_adv_loss = binary_cross_entropy(score, labels)
            # Contrastive motion loss
            encoder.to(device)
            cml = contrastive_motion_loss(stable_sequence,
                                            unstable_sequence,
                                            generated_sequence)
            g_loss += generator_adv_loss + 10 * cml 
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            g_running_loss += g_loss.item()
            d_running_loss += d_loss.item()
            print(f'\repoch: {epoch},batch: {idx}, generator_loss:{g_running_loss / (idx % 1000 + 1)} ,per: {percept_loss.item()}, cx: {context_loss.item()},\
                    cml: {cml.item()},adv: {generator_adv_loss.item()} , discriminator_loss:{d_loss.item():.3f}',end = '')
            del feat1, feat2, percept_loss, context_loss
            
            #visualization
            means = np.array([0.155,0.161,0.153],dtype = np.float32)
            stds = np.array([0.22,0.231,0.226],dtype = np.float32)
            img = generated_sequence[:,:,0,:,:].permute(0,2,3,1)[0,...].cpu().detach().numpy()
            img *= stds
            img += means
            img = np.clip(img * 255.0,0,255).astype(np.uint8)
            img1 = stable_sequence[:,:,0,:,:].permute(0,2,3,1)[0,...].cpu().numpy()
            img1 *= stds
            img1 += means
            img1 = np.clip(img1 * 255.0,0,255).astype(np.uint8)
            concat = cv2.hconcat([img,img1])
            cv2.imshow('window',concat)
            if cv2.waitKey(1) & 0xFF == ord('9'):
                break
            if idx % 1000 == 999:
                writer.add_scalar('generator_loss',
                                    g_running_loss / 1000,
                                    epoch * dataset_len + idx)
                writer.add_scalar('discriminator_loss',
                                    d_running_loss / 1000,
                                    epoch * dataset_len + idx)
                g_running_loss = 0.0
                d_running_loss = 0.0
                model_path = os.path.join(args.ckpt_dir,'generator/',f'generator_{epoch}.pth')
                torch.save({'model':generator.state_dict(),
                            'optimizer' : g_optimizer.state_dict(),
                            'epoch' : epoch}
                        ,model_path)
                model_path = os.path.join(args.ckpt_dir,'discriminator/',f'discriminator_{epoch}.pth')
                torch.save({'model':discriminator.state_dict(),
                            'optimizer' : d_optimizer.state_dict(),
                            'epoch' : epoch}
                        ,model_path)

    cv2.destroyAllWindows()



if __name__ == '__main__':
    device = 'cuda'
    starting_epoch = 0
    args = parse_args()
    #set up models
    if args.model == 'ENet':
        generator = ENet(in_channels=15, out_channels=3, residual_blocks=64).train().to(device)
    elif args.model == 'UNet':
        generator = UNet().train().to(device)
    else:
        print('Please choose either ENet or UNet model')
        exit()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr= 1e-5)
    discriminator = TempDiscriminator3D(d=32).train().to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5,betas=(0.9, 0.9))
    encoder = models.video.mc3_18(weights='MC3_18_Weights.KINETICS400_V1')
    encoder = nn.Sequential(*list(encoder.children())[:-1][:-1]).to(device).eval() #131072 output vector for 1,3,8,256,256
    mobile = models.mobilenet_v3_small(weights=True)
    mobile = mobile.features.eval().to(device)
    #load checkpoints
    load_checkpoint(generator, discriminator,g_optimizer,d_optimizer, args.ckpt_dir)
    #load dataset
    train_ds = get_data_loader(shape = args.shape, txt_path = args.txt_path, skip = SKIP, batch_size=args.batch_size)
    writer = SummaryWriter('./runs/stage3/')
    train(generator, g_optimizer, discriminator, d_optimizer,train_ds, writer, starting_epoch, args)