import pickle
import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='create txt with training data')
parser.add_argument('--src_stable_path', type=str, help='path to original DeepStab stable videos')
parser.add_argument('--src_unstable_path', type=str, help='path to original DeepStab unstable videos')
parser.add_argument('--dst_stable_path', type=str, help='path to cropped stable videos')
parser.add_argument('--dst_unstable_path', type=str, help='path to cropped unstable videos')
args = parser.parse_args()

# Check if destination paths exist, if not, create them
os.makedirs(args.dst_stable_path, exist_ok=True)
os.makedirs(args.dst_unstable_path, exist_ok=True)

stable_path = args.src_stable_path
unstable_path = args.src_unstable_path
dst_stable_path = args.dst_stable_path
dst_unstable_path = args.dst_unstable_path
with open('./valid_sequences_dict.pkl', 'rb') as file:
    sequences_dict = pickle.load(file)

h,w = 720, 1280
th, tw = int(0.75 * h), int(0.75 * w)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videos = os.listdir(stable_path)
video_idx = 1
videos = os.listdir(stable_path)
for video_name in videos:
    print(video_name)
    sequences, pos = sequences_dict[video_name]
    for i, seq in enumerate(sequences):
        s_cap = cv2.VideoCapture(stable_path + video_name)
        u_cap = cv2.VideoCapture(unstable_path + video_name)
        start_idx, end_idx = seq 
        s_cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        u_cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        s_out = cv2.VideoWriter(dst_stable_path + f'{video_idx}.avi', fourcc, 30.0, (tw, th))
        u_out = cv2.VideoWriter(dst_unstable_path + f'{video_idx}.avi', fourcc, 30.0, (tw, th))
        video_idx += 1
        for el,frame_idx in enumerate(range(start_idx,end_idx+1)):
            ret1,stable_frame = s_cap.read()
            ret2,unstable_frame = u_cap.read()
            if not ret1 or not ret2:
                print(video_name,frame_idx, 'broken')
                break
            x = pos[i][el][0]
            y = pos[i][el][1]
            stable_cropped = stable_frame[y: y+th, x: x+tw, :]
            unstable_cropped = unstable_frame[y: y+th, x: x+tw, :]
            s_out.write(stable_cropped)
            u_out.write(unstable_cropped)
        s_cap.release()
        u_cap.release()
        s_out.release()
        u_out.release()