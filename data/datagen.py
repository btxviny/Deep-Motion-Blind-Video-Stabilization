import numpy as np
import glob
import cv2
import os
import random
import torch

torch.manual_seed(42)
from torchvision import transforms
Augmentation = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(0.4),
    transforms.RandomVerticalFlip(0.1),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, hue = 0),
    transforms.Normalize(mean=[0.155,0.161,0.153], std=[0.228,0.231,0.226])
)


class DataGenerator:
    def __init__(self, shape, txt_path, skip = 16):
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
            ret1,Igt = stable_cap.read()
            if not ret1 : continue
            Igt = resize(Igt, self.shape)
            
            unstable_cap = cv2.VideoCapture(u_path)
            unstable_frames = np.zeros((5,h,w,3),dtype=np.float32)
            for i,pos in enumerate(range(frame_idx - SKIP, frame_idx + SKIP + 1, SKIP // 2)):
                unstable_cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
                ret2,frame = unstable_cap.read()
                if not ret2: 
                    flag = False
                    break
                frame = resize(frame, self.shape)
                unstable_frames[i,...] = frame
                
            if not flag:
                continue
            unstable_frames = torch.from_numpy(unstable_frames).permute(0,-1,1,2).float()
            Igt = torch.from_numpy(Igt).permute(2,0,1).float()
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            unstable_frames = Augmentation(unstable_frames)
            torch.manual_seed(seed)
            Igt = Augmentation(Igt)
            unstable_frames = unstable_frames.reshape(-1,h,w)
            yield unstable_frames, Igt

def resize(img,shape):
    h,w,_ = shape
    img = cv2.resize(img,(w,h),cv2.INTER_LINEAR)
    img = img / 255.0
    return img