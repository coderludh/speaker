import cv2
import os
from tqdm import tqdm

# 输入和输出文件夹路径
input_folder = r'E:\python_project\speaker\tem\videos'  # 替换为你的输入文件夹路径
output_folder = r'E:\python_project\speaker\tem\videos\frame'  # 替换为你的输出文件夹路径

# 目标分辨率
target_width, target_height = 320, 240

# 支持的视频文件扩展名
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有视频文件列表
video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]

if not video_files:
    print("输入文件夹中没有找到支持的视频文件。")
    exit()

print(f"找到 {len(video_files)} 个视频文件。开始处理...")

for video_file in tqdm(video_files, desc="处理视频"):
    input_video_path = os.path.join(input_folder, video_file)

    # 构建输出视频路径，保留原文件名
    output_video_path = os.path.join(output_folder, video_file)

    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_file}")
        continue

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取视频的编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以根据需要更改编解码器

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 调整帧的大小
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # 写入调整后的帧
        out.write(resized_frame)

    # 释放资源
    cap.release()
    out.release()

print(f"所有视频已成功处理并保存在 '{output_folder}' 文件夹中。")
