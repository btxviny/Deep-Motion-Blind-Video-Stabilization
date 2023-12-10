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
import torchvision

from models.models import ENet,UNet,Critic
from data.datagen import DataGenerator
from config import SKIP

def parse_args():
    parser = argparse.ArgumentParser(description='DMBVS-stage1-train')
    parser.add_argument('--model', type=str, help='Model Architecture')
    parser.add_argument('--ckpt_dir', type=str, help='path to stage 1 weights directory')
    parser.add_argument('--shape', nargs='+', type=int, help='H W C values', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--txt_path', type=str, help='path to training list')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='Lambda for gradient penalty term')
    parser.add_argument('--n_critic', type=int, default=5, help='Number of critic updates per generator update')
    return parser.parse_args()

def load_checkpoint(generator,critic,g_optimizer,c_optimizer, ckpt_dir):
    #check if directory exists, if not load weights from previous session
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        os.makedirs(os.path.join(ckpt_dir,'generator/'))
        os.makedirs(os.path.join(ckpt_dir,'critic/'))
        prev_ckpt_dir = './stage1/'
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
        critic_path = os.path.join(ckpt_dir,'critic/')
        ckpts = [x for x in os.listdir(critic_path) if x.endswith('.pth')]
        ckpts = sorted(ckpts, key=lambda x: int(x.split('.')[0].split('_')[1]))  # sort by epoch number
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(critic_path, latest))
        critic.load_state_dict(state_dict['model'])
        c_optimizer.load_state_dict(state_dict['optimizer'])
        print(f'Loaded critic: {latest}')

def get_data_loader(shape, txt_path, skip, batch_size):
    class IterDataset(data.IterableDataset):
        def __init__(self, data_generator):
            super(IterDataset, self).__init__()
            self.data_generator = data_generator
        def __iter__(self):
            return iter(self.data_generator())
    data_gen = DataGenerator(shape = shape, txt_path=txt_path, skip = skip)
    data_gen = IterDataset(data_gen)
    train_ds = data.DataLoader(data_gen, batch_size = batch_size)
    return train_ds

def preprocess_img(img):
    
    normalize = torchvision.transforms.Normalize(mean=[0.155,0.161,0.153], std=[0.228,0.231,0.226])

    # Convert the image to a tensor and normalize it
    img = normalize(img)
    return img

def perceptual_loss(img1, img2):
    b,c,h,w = img1.shape
    img2.to(img1.device)
    img1_tensor = preprocess_img(img1)
    img2_tensor = preprocess_img(img2)

    x = vgg19(img1_tensor)
    x_norm = x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    y = vgg19(img2_tensor)
    y_norm = y / torch.sqrt(torch.sum(y**2, dim=1, keepdim=True))
    return torch.sqrt(torch.sum((x_norm - y_norm)**2)) ** 2 / (c*h*w)

def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones_like(d_interpolates, device=real_samples.device)
    gradient = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm -1) ** 2)
    return gradient_penalty


def train(generator, g_optimizer, critic, c_optimizer,train_ds, writer, starting_epoch, args):
    torch.cuda.empty_cache()
    EPOCHS = 10
    device = 'cuda'
    dataset_len = 50000
    running_g_loss = 0.0
    running_c_loss = 0.0
    cv2.namedWindow('stage2', cv2.WINDOW_NORMAL)
    for epoch in range(starting_epoch, EPOCHS):
        for idx, batch in enumerate(train_ds):
            sequence,Igt = batch
            sequence = sequence.to(device)
            Igt = Igt.to(device)

            if idx > 0 and idx % args.n_critic == 0: #train both models
                #train
                c_optimizer.zero_grad()
                # Generate a frame and its corresponding transformation
                #freeze generator
                with torch.no_grad():
                    generated_frame = generator(sequence)
                # Compute the critic scores for real and generated frames
                real_scores = critic(Igt)
                fake_scores = critic(generated_frame.detach())  # Detach to avoid generator update
                critic_loss = -(torch.mean(real_scores) - torch.mean(fake_scores))
                gradient_penalty = compute_gradient_penalty(critic, Igt, generated_frame)
                critic_loss += args.lambda_gp * gradient_penalty
                critic_loss.backward()
                c_optimizer.step()
                
                g_optimizer.zero_grad()
                generated_frame = generator(sequence)
                with torch.no_grad():
                    gen_scores = critic(generated_frame)

                percept_loss = perceptual_loss(generated_frame, Igt) 
                adv_loss = -torch.mean(gen_scores.detach())
                gen_loss = 1000 * percept_loss  + 0.1 * adv_loss
                # Backpropagate and optimize the generator
                gen_loss.backward()
                g_optimizer.step()


            else: #train only critic
                #train
                c_optimizer.zero_grad()
                # Generate a frame and its corresponding transformation
                #freeze generator
                with torch.no_grad():
                    generated_frame = generator(sequence)
                # Compute the critic scores for real and generated frames
                real_scores = critic(Igt)
                fake_scores = critic(generated_frame.detach())  # Detach to avoid generator update
                critic_loss = -(torch.mean(real_scores) - torch.mean(fake_scores))
                gradient_penalty = compute_gradient_penalty(critic, Igt, generated_frame)
                critic_loss += args.lambda_gp * gradient_penalty
                critic_loss.backward()
                c_optimizer.step()
                
            
            means = np.array([0.155,0.161,0.153],dtype = np.float32)
            stds = np.array([0.22,0.231,0.226],dtype = np.float32)
            img = generated_frame.permute(0,2,3,1)[0,...].cpu().detach().numpy()
            img *= stds
            img += means
            img = np.clip(img * 255.0,0,255).astype(np.uint8)
            img1 = Igt.permute(0,2,3,1)[0,...].cpu().detach().numpy()
            img1 *= stds
            img1 += means
            img1 = np.clip(img1 * 255.0,0,255).astype(np.uint8)
            conc = cv2.hconcat([img,img1])
            cv2.imshow('stage2',conc)
            if cv2.waitKey(1) & 0xFF == ord('9'):
                break
            if idx < args.n_critic :
                print(f'\repoch: {epoch}, batch: {idx}, critic_loss: {critic_loss.item()}',end = '')
                
            else:
                print(f'\repoch: {epoch}, batch: {idx}, perceptual: {1000 *percept_loss.item()}, adv_loss: {0.1* adv_loss.item()},\
                    critic_loss: {critic_loss.item()}',end = '')
                running_g_loss += gen_loss.item()
            running_c_loss += critic_loss.item()
            #save weights
            if idx % 1000 == 999:
                writer.add_scalar('generator_loss',
                                running_g_loss / 1000,
                                epoch * 50000 + idx)
                writer.add_scalar('critic_loss',
                                running_c_loss / 1000,
                                epoch * 50000 + idx)
                running_g_loss = 0.0
                running_c_loss = 0.0
                model_path = os.path.join(args.ckpt_dir,'generator/',f'generator_{epoch}.pth')
                torch.save({'model':generator.state_dict(),
                            'optimizer' : g_optimizer.state_dict(),
                            'epoch' : epoch}
                        ,model_path)
                model_path = os.path.join(args.ckpt_dir,'critic/',f'critic_{epoch}.pth')
                torch.save({'model':critic.state_dict(),
                            'optimizer' : c_optimizer.state_dict(),
                            'epoch' : epoch}
                        ,model_path)
            


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
    g_optimizer = torch.optim.Adam(generator.parameters(), lr= 5e-5)
    critic = Critic(d=64).train().to(device)
    c_optimizer = torch.optim.Adam(critic.parameters(), lr = 5e-5,betas=(0, 0.9))
    vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1')
    vgg19 = nn.Sequential(*list(vgg19.children())[0][:-1]) # use all layers up to relu3_3
    vgg19.eval().to(device)
    #load checkpoints
    load_checkpoint(generator,critic,g_optimizer,c_optimizer, args.ckpt_dir)
    #load dataset
    train_ds = get_data_loader(shape = args.shape, txt_path = args.txt_path, skip = SKIP, batch_size=args.batch_size)
    writer = SummaryWriter('./runs/stage2/')
    train(generator, g_optimizer, critic, c_optimizer,train_ds, writer, starting_epoch, args)