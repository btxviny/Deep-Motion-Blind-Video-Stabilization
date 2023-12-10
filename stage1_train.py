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

from models.models import ENet,UNet
from data.datagen import DataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='DMBVS-stage1-train')
    parser.add_argument('--model', type=str, help='Model Architecture')
    parser.add_argument('--ckpt_dir', type=str, help='path to stage 1 weights directory')
    parser.add_argument('--shape', nargs='+', type=int, help='H W C values', required=True)
    parser.add_argument('--skip', type=int, help='temporal distance of input frames')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--txt_path', type=str, help='path to training list')
    return parser.parse_args()

def load_checkpoint(model, optimizer, ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f'Checkpoint directory {ckpt_dir} created.')
    ckpts = [x for x in os.listdir(ckpt_dir) if x.endswith('.pth')]
    if ckpts:
        ckpts = sorted(ckpts, key=lambda x: int(x.split('.')[0].split('_')[1]))  # sort by epoch number
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(ckpt_dir, latest))
        model.load_state_dict(state_dict['model'])
        starting_epoch = state_dict['epoch'] + 1
        optimizer.load_state_dict(state_dict['optimizer'])
        print('Loaded weights from the previous session')
        print(f'Starting from epoch {starting_epoch}')
    else:
        print(f'No checkpoint files found in {ckpt_dir}.')


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

def train(generator, optimizer, train_ds, writer, starting_epoch):
    
    cv2.namedWindow('stage2', cv2.WINDOW_NORMAL)
    EPOCHS = 10
    dataset_len = 50000
    for epoch in range(starting_epoch, EPOCHS):
        running_loss = 0
        for idx, batch in enumerate(train_ds):
            sequence, Igt = batch
            sequence = sequence.to('cuda')
            Igt = Igt.to('cuda')

            generated_frame = generator(sequence)
            optimizer.zero_grad()
            loss = F.mse_loss(generated_frame, Igt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            means = np.array([0.155, 0.161, 0.153], dtype=np.float32)
            stds = np.array([0.228, 0.231, 0.226], dtype=np.float32)
            img = generated_frame.permute(0, 2, 3, 1)[0, ...].cpu().detach().numpy()
            img *= stds
            img += means
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img1 = Igt.permute(0, 2, 3, 1)[0, ...].cpu().detach().numpy()
            img1 *= stds
            img1 += means
            img1 = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
            conc = cv2.hconcat([img, img1])
            cv2.imshow('stage2', conc)
            if cv2.waitKey(1) & 0xFF == ord('9'):
                break
            print(f'\repoch: {epoch}, batch: {idx}, loss: {running_loss / (idx % 1000 + 1)}', end='')
            if idx % 1000 == 999:
                writer.add_scalar('loss',
                                  running_loss / 1000,
                                  epoch * dataset_len + idx)
                running_loss = 0
                model_path = os.path.join(args.ckpt_dir, f'generator_{epoch}.pth')
                torch.save({'model': generator.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch}
                           , model_path)

if __name__ == '__main__':
    device = 'cuda'
    starting_epoch = 0
    args = parse_args()
    if args.model == 'ENet':
        generator = ENet(in_channels=15, out_channels=3, residual_blocks=64).train().to(device)
    elif args.model == 'UNet':
        generator = UNet().train().to(device)
    else:
        print('Please choose either ENet or UNet model')
        exit()
    optimizer = torch.optim.Adam(generator.parameters(), lr= 1e-4)
    load_checkpoint(generator, optimizer, args.ckpt_dir)
    train_ds = get_data_loader(shape = args.shape, txt_path = args.txt_path, skip = args.skip, batch_size=args.batch_size)
    writer = SummaryWriter('./runs/stage1/')
    train(generator, optimizer, train_ds, writer, starting_epoch)