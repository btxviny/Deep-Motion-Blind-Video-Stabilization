# Deep-Motion-Blind-Video-Stabilization

This is a PyTorch implementation of the paper [Deep Motion Blind Video Stabilization](https://arxiv.org/abs/2011.09697).

![Video Stabilization Example](https://github.com/btxviny/Deep-Motion-Blind-Video-Stabilization/blob/main/result.gif)

## Inference Instructions

Follow these instructions to perform video stabilization using the pretrained model:

1. **Download Pretrained Model:**
   - Download the pretrained model [weights](https://drive.google.com/file/d/1zi5ASOnSdWRxrtIzz16WfOi3maB5Nylm/view?usp=drive_link).
   - Place the downloaded weights file in the main folder of your project.

2. **Run the Stabilization Script:**
   - Open a terminal and navigate to the main folder of your project.
   - Run the following command:
     ```bash
     python stabilize.py --model ENet --in_path unstable_video_path --out_path result_path
     ```
   - Replace `unstable_video_path` with the path to your input unstable video.
   - Replace `result_path` with the desired path for the stabilized output video.

Make sure you have the necessary dependencies installed and that your environment is set up correctly before running the command.

Feel free to customize the paths and instructions based on your project's structure. If you have any additional details or variations in the process, you can include them in these instructions.
