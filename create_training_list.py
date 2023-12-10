import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='create txt with training data')
parser.add_argument('--stable_path', type=str, help='path to stable videos of DeepStab Modded')
parser.add_argument('--unstable_path', type=str, help='path to unstable videos of DeepStab Modded')
parser.add_argument('--skip', type=int, help='temporal distance of input frames')
parser.add_argument('--txt_path', type=str, help='path to create trainlist txt file')
# Parse the command-line arguments
args = parser.parse_args()

txt_path = args.txt_path
stable_path = args.stable_path
unstable_path = args.unstable_path
video_names = os.listdir(unstable_path)
with open(txt_path,'w') as f:    
    for video in video_names:
        s_path = os.path.join(stable_path,video)
        u_path = os.path.join(unstable_path,video)
        stable_cap = cv2.VideoCapture(s_path)
        unstable_cap = cv2.VideoCapture(u_path)
        s_frame_count = int(stable_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        u_frame_count = int(unstable_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_of_frames = min(s_frame_count,u_frame_count)
        
        for frame_idx in range(args.skip,num_of_frames- args.skip):
            line = f'{s_path},{u_path},{frame_idx}\n'
            f.write(line)

'''txt_path = f'./trainlist_stage3_(+-{SKIP}).txt'
stable_path = 'E:/Datasets/DeepStab cropped/stable'
unstable_path = 'E:/Datasets/DeepStab cropped/unstable'
video_names = os.listdir(unstable_path)
with open(txt_path,'w') as f:    
    for video in video_names:
        s_path = os.path.join(stable_path,video)
        u_path = os.path.join(unstable_path,video)
        stable_cap = cv2.VideoCapture(s_path)
        unstable_cap = cv2.VideoCapture(u_path)
        s_frame_count = int(stable_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        u_frame_count = int(unstable_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_of_frames = min(s_frame_count,u_frame_count)
        
        for frame_idx in range(SKIP,num_of_frames - 8 * SKIP):
            line = f'{s_path},{u_path},{frame_idx}\n'
            f.write(line)'''
