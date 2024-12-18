# video_processor.py
import json
import cv2
import mediapipe as mp
import numpy as np
from common import *
import torch
from PIL import Image
import ffmpeg
import os
import subprocess
from FaceNet.common import *
from recorder import VideoRecorder


class VideoProcessor:
    def __init__(self, model_path, similarity_threshold=0.5, top_k=2, speaking_threshold=5):
        """
        初始化视频处理器。

        :param model_path: 人脸识别模型路径
        :param similarity_threshold: 相似度阈值
        :param top_k: 返回相似度最高的k个结果
        :param speaking_threshold: 判断是否在说话的唇部移动阈值
        """
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,  # 支持多达10个人
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 加载人脸识别模型
        self.model = load_recognition_model(model_path)
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

        # 唇部关键点索引（MediaPipe Face Mesh 中唇部的关键点）
        self.lips_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
                             324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
                             37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13,
                             82, 81, 80, 191, 78, 95]

        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.speaking_threshold = speaking_threshold

        # 初始化视频录制器
        self.recorder = VideoRecorder()
        self.recorder.start()

    def is_speaking(self, landmarks, frame_width, frame_height, previous_distance, threshold=5):
        """
        判断唇部是否有显著移动来推断是否在说话。
        """
        # 计算上下唇之间的距离
        upper_lip = np.array([landmarks[13][1] * frame_width, landmarks[13][0] * frame_height])
        lower_lip = np.array([landmarks[14][1] * frame_width, landmarks[14][0] * frame_height])
        distance = np.linalg.norm(upper_lip - lower_lip)

        if previous_distance is None:
            return False, distance
        movement = abs(distance - previous_distance)
        speaking = movement > threshold
        return speaking, distance

    def identify_person(self, aligned_face):
        """
        识别身份，返回前两个相似的特征。
        """
        imgs = preprocess_image(aligned_face)
        feature = extract_feature(imgs, self.model)
        top_matches = compare_face_features(feature, device=self.device, top_k=self.top_k)
        return top_matches

    def merge_video_segments(self, T_start, T_end, log_file='tem/segments_log.json', output_dir='tem/videos',
                             ):
        if not os.path.exists(log_file):
            print("日志文件不存在。")
            return False

        s_str = int(T_start)
        e_str = int(T_end)
        output_file = f"tem/videos/merged/{s_str}_{e_str}.avi"

        with open(log_file, 'r') as f:
            segments = json.load(f)

        # 查找重叠视频段
        overlapping_segments = []
        for entry in segments:
            seg_s = entry['start_time']
            seg_e = entry['end_time']
            if seg_e >= T_start and seg_s <= T_end:
                video_path = os.path.join(output_dir, entry['filename'])
                if os.path.exists(video_path):
                    overlapping_segments.append((seg_s, seg_e, video_path))

        if not overlapping_segments:
            print("给定时间区间内无视频数据。")
            return False

        # 打开第一个视频获取参数（例如分辨率、fps、编码格式）
        first_seg_path = overlapping_segments[0][2]
        cap_init = cv2.VideoCapture(first_seg_path)
        if not cap_init.isOpened():
            print("无法打开首个视频。")
            return False

        width = int(cap_init.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_init.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_init.get(cv2.CAP_PROP_FPS)
        cap_init.release()

        # 初始化输出VideoWriter（使用XVID编码器为例）
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        if not out.isOpened():
            print("无法创建输出文件。")
            return False

        total_frames_written = 0

        for (seg_s, seg_e, video_path) in overlapping_segments:
            analysis_start = max(seg_s, T_start)
            analysis_end = min(seg_e, T_end)
            if analysis_end <= analysis_start:
                continue  # 无有效区间

            # 计算需要跳过和读取的帧数
            skip_seconds = analysis_start - seg_s
            analysis_duration = analysis_end - analysis_start
            skip_frames = int(skip_seconds * fps)
            frames_to_analyze = int(analysis_duration * fps)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                continue

            # 跳过不需要的帧
            for _ in range(skip_frames):
                ret = cap.read()
                if not ret:
                    break

            # 写入需要的帧
            frames_written_this_segment = 0
            for _ in range(frames_to_analyze):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written_this_segment += 1

            total_frames_written += frames_written_this_segment
            cap.release()

        out.release()

        if total_frames_written == 0:
            print("未能写入任何帧，请检查时间区间和视频段是否匹配。")
            return False

        print(f"成功生成合并视频：{output_file}，共写入{total_frames_written}帧。")
        return True



