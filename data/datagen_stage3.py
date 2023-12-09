import numpy as np
import glob
import cv2
import os
import random
import torch

torch.manual_seed(42)
from torchvision import transforms
Augmentation = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.ColorJitter(brightness = 0.4, contrast = 0.4, hue = 0.1),
    transforms.Normalize(mean=[0.155,0.161,0.153], std=[0.228,0.231,0.226])
)


class DataGenerator:
    def __init__(self, shape, txt_path = './trainlist_stage3(+-16).txt', skip = 16):
        self.shape = shape
        self.skip = skip
        with open(txt_path, 'r') as f:
            self.trainlist = f.read().splitlines()
        
    def __call__(self):
        h,w,c = self.shape
        SKIP = self.skip
        self.trainlist = random.sample(self.trainlist, len(self.trainlist))
        for sample in self.trainlist:
            flag = True
            s_path = sample.split(',')[0]
            u_path = sample.split(',')[1]
            frame_idx = int(sample.split(',')[2])

            stable_cap = cv2.VideoCapture(s_path)
            stable_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            Igt = []
            for idx in range(8):
                ret1,img = stable_cap.read()
                if not ret1 : continue
                Igt.append(resize(img, self.shape))
            
            unstable_cap = cv2.VideoCapture(u_path)
            unstable_frames = np.zeros((5,h,w,3),dtype=np.float32)
            unstable_sequence = []
            for idx in range(frame_idx, frame_idx + 8):
                for i,pos in enumerate(range(idx - SKIP, idx + SKIP + 1, SKIP // 2)):
                    unstable_cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
                    ret2,frame = unstable_cap.read()
                    if not ret2: 
                        flag = False
                        break
                    frame = resize(frame, self.shape)
                    unstable_frames[i,...] = frame
                unstable_sequence.append(unstable_frames)
                
            if not flag:
                continue
            #set the same seed for all augmentation transforms
            seed = random.randint(0, 2**32)
            input_sequence_augmented = []
            unstable_sequence_augmented = torch.zeros(1,3,8,h,w)
            for i,u in enumerate(unstable_sequence):
                u = torch.from_numpy(u).permute(0,-1,1,2).float()
                torch.manual_seed(seed)
                u = Augmentation(u)
                unstable_sequence_augmented[:,:,i,:,:] = u[2,...].unsqueeze(0)
                input_sequence_augmented.append(u.reshape(-1,h,w).unsqueeze(0))

            igt_augmented = torch.zeros(1,3,8,h,w)
            for i,igt in enumerate(Igt):
                igt = torch.from_numpy(igt).permute(2,0,1).float()
                torch.manual_seed(seed)
                igt = Augmentation(igt)
                igt_augmented[:,:,i,:,:] = igt.unsqueeze(0)
            
            yield input_sequence_augmented, unstable_sequence_augmented, igt_augmented

def resize(img,shape):
    h,w,_ = shape
    img = cv2.resize(img,(w,h),cv2.INTER_LINEAR)
    img = img / 255.0
    return img