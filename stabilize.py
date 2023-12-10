import numpy as np
import cv2
import argparse
import os
import torch
from models.models import ENet,UNet

device = 'cuda'
H,W = 256,256


def parse_args():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--model', type=str, help='Model Architecture to use')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    return parser.parse_args()


def save_video(frames, path):
    frame_count,h,w,_ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w,h))
    for idx in range(frame_count):
        out.write(frames[idx,...])
    out.release()

    
def stabilize(in_path,out_path,skip):
    
    if not os.path.exists(in_path):
        print(f"The input file '{in_path}' does not exist.")
        exit()
    _,ext = os.path.splitext(in_path)
    if ext not in ['.mp4','.avi']:
        print(f"The input file '{in_path}' is not a supported video file (only .mp4 and .avi are supported).")
        exit()

    #Load frames and stardardize
    cap = cv2.VideoCapture(in_path)
    mean = np.array([0.155,0.161,0.153],dtype = np.float32) 
    std = np.array([0.228,0.231,0.226],dtype = np.float32)
    frames = []
    while True:
        ret, img = cap.read()
        if not ret: break
        img = cv2.resize(img, (W,H))
        img = (img / 255.0).astype(np.float32)
        img = (img - mean)/std
        frames.append(img)
    frames = np.array(frames,dtype = np.float32)
    frame_count,_,_,_ = frames.shape
    
    # stabilize video
    frames_tensor = torch.from_numpy(frames).permute(0,3,1,2).float().to('cpu')
    stable_frames_tensor = frames_tensor.clone()
    SKIP = skip
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    def get_batch(idx):
        batch = torch.zeros((5,3,H,W)).float()
        for i,j in enumerate(range(idx - SKIP, idx + SKIP + 1, SKIP//2)):
                batch[i,...] = frames_tensor[j,...]
        batch = batch.view(1,-1,H,W)
        return batch.to(device)

    for frame_idx in range(SKIP,frame_count - SKIP):
        batch = get_batch(frame_idx)
        with torch.no_grad():
            generated_frame = model(batch.cuda())
        stable_frames_tensor[frame_idx,...] = generated_frame
        # Write the stabilized frame to the output video file
        img = generated_frame.permute(0,2,3,1)[0,...].cpu().detach().numpy()
        img *= std
        img += mean
        img = np.clip(img * 255.0,0,255).astype(np.uint8)
        # Display the stabilized frame (optional)
        cv2.imshow('window', img)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    #undo standardization
    stable_frames = np.clip(((stable_frames_tensor.permute(0,2,3,1).numpy() * std) + mean) * 255,0,255).astype(np.uint8)
    save_video(stable_frames,out_path)


if __name__ == '__main__':
    args = parse_args()
    if args.model == 'ENet':
        model = ENet(in_channels=15, out_channels=3, residual_blocks=64).train().to(device)
        skip = 2
        ckpt_dir = './ckpts/ENet/'
    elif args.model == 'UNet':
        model = UNet().train().to(device)
        skip = 16
        ckpt_dir = './ckpts/UNet/'
    else:
        print('Please choose either ENet or UNet model')
        exit()
    ckpts = [x for x in os.listdir(ckpt_dir) if x.endswith('.pth')]
    if ckpts:
        ckpts = sorted(ckpts, key = lambda x : x.split('.')[0].split('_')[1]) #sort
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(ckpt_dir,latest))
        model.load_state_dict(state_dict['model'])
        print(f'loaded weights from previous session: {latest}')
    stabilize(args.in_path, args.out_path,skip)
