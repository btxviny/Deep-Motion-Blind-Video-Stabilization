# Deep-Motion-Blind-Video-Stabilization

This is a PyTorch implementation of the paper [Deep Motion Blind Video Stabilization](https://arxiv.org/abs/2011.09697).

![Video Stabilization Example](https://github.com/btxviny/Deep-Motion-Blind-Video-Stabilization/blob/main/result.gif)

## Inference Instructions

Follow these instructions to perform video stabilization using the pretrained model:

1. **Download Pretrained Model:**
   - Download the pretrained model [weights](https://drive.google.com/file/d/1zi5ASOnSdWRxrtIzz16WfOi3maB5Nylm/view?usp=drive_link).
   - Place the downloaded weights file in the main folder of your project.

2. **Run the Stabilization Script:**
   - Run the following command:
     ```bash
     python stabilize.py --model `ENet` --in_path unstable_video_path --out_path result_path
     ```
   - Replace `unstable_video_path` with the path to your input unstable video.
   - Replace `result_path` with the desired path for the stabilized output video.
   - You can choose between --model `ENet` and `UNet`.

Make sure you have the necessary dependencies installed, and that your environment is set up correctly before running the command.

## Training Instructions

Follow these instructions to train the model:

1. **Download Datasets:**
   - Download the training datasets:
     - [DeepStab Modded](https://hyu-my.sharepoint.com/personal/kashifali_hanyang_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkashifali%5Fhanyang%5Fac%5Fkr%2FDocuments%2FDeepStab%5FMod%2Erar&parent=%2Fpersonal%2Fkashifali%5Fhanyang%5Fac%5Fkr%2FDocuments&ga=1)
     - [DeepStab](https://cg.cs.tsinghua.edu.cn/people/~miao/stabnet/demo.zip)

   - Extract the contents of the downloaded datasets to a location on your machine.

2. **Create Dataset for Third Training Stage:**
   - Run the following command to create the dataset for the third training stage:

     ```bash
     python create_dataset_stage3.py --src_stable_path /path/to/deepstab_stable_videos --src_unstable_path /path/to/deepstab_unstable_videos --dst_stable_path /path/to/generated_stable_videos --dst_unstable_path /path/to/generated_unstable_videos
     ```

     - Adjust `/path/to/deepstab_stable_videos`, `/path/to/deepstab_unstable_videos`, `/path/to/generated_stable_videos`, and `/path/to/generated_unstable_videos` with the actual paths for your project.
