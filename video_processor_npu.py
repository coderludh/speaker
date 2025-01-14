# video_processor.py
import json
import cv2
import mediapipe as mp
import numpy as np
from torch.utils.tensorboard.summary import video, image
import math
from common_npu import *
import torch
from PIL import Image
import ffmpeg
import os
import subprocess
from recorder import VideoRecorder
import logging
from collections import defaultdict

from common import *

class VideoProcessorNPU:
    def __init__(self, rknn_path=r'./FaceNet/models/mobilefacenet.rknn', similarity_threshold=0.5, top_k=2, mouth_open_threshold=5.0, face_tracking_threshold=50):
        """
        初始化视频处理器。

        :param model_path: 人脸识别模型路径
        :param similarity_threshold: 相似度阈值
        :param top_k: 返回相似度最高的k个结果

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
        self.mouth_open_threshold = mouth_open_threshold
        self.face_tracking_threshold = face_tracking_threshold
        # 加载人脸识别模型
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.rknn = RKNNLite()
        self.ret = self.rknn.load_rknn(path=rknn_path)
        self.ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

        # 初始化视频录制器
        # self.recorder = VideoRecorder()
        # self.recorder.start()

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
        feature = extract_feature(imgs, self.ret)

        top_matches = compare_face_features(feature, device=self.device, top_k=self.top_k)
        return top_matches


    def merge_video_segments(self, T_start, T_end, log_file='tem/segments_log.json', output_dir='tem/videos/merged'):
        """
        合并指定时间范围内的视频片段，包括memory_buffer中的帧。
        :param T_start: 开始时间（秒，浮点数）
        :param T_end: 结束时间（秒，浮点数）
        :param log_file: 视频片段日志文件路径
        :param output_dir: 合并后视频的输出目录
        :return: 合并后的视频文件路径或False
        """
        try:
            # 设置日志配置
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

            if not os.path.exists(log_file):
                logging.error("日志文件不存在。")
                return False

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 保持时间为浮点数，避免精度丢失
            s_str = float(T_start)
            e_str = float(T_end)
            output_file = os.path.join(output_dir, f"{int(s_str)}_{int(e_str)}.mp4")

            with open(log_file, 'r') as f:
                segments = json.load(f)

            # 查找重叠视频段
            overlapping_segments = []
            for entry in segments:
                seg_s = float(entry['start_time'])
                seg_e = float(entry['end_time'])
                if seg_e >= T_start and seg_s <= T_end:
                    video_path = os.path.join('tem/videos', entry['filename'])
                    if os.path.exists(video_path):
                        overlapping_segments.append((seg_s, seg_e, video_path))
                    else:
                        logging.warning(f"找不到视频文件: {video_path}")

            if not overlapping_segments:
                logging.warning("给定时间区间内无视频数据。")
                return False

            # 按开始时间排序
            overlapping_segments.sort(key=lambda x: x[0])

            # 打开第一个视频获取参数（例如分辨率、fps、编码格式）
            first_seg_path = overlapping_segments[0][2]
            cap_init = cv2.VideoCapture(first_seg_path)
            if not cap_init.isOpened():
                logging.error("无法打开首个视频。")
                return False

            width = int(cap_init.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_init.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap_init.get(cv2.CAP_PROP_FPS)
            cap_init.release()

            # 初始化输出VideoWriter（使用mp4v编码器）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            if not out.isOpened():
                logging.error("无法创建输出文件。")
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
                    logging.warning(f"无法打开视频: {video_path}")
                    continue

                # 跳过不需要的帧
                for _ in range(skip_frames):
                    ret, _ = cap.read()
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

            # 处理memory_buffer中的帧
            recent_frames = self.recorder.get_recent_memory_buffer(T_end)
            for frame in recent_frames:
                # 假设frame是一个numpy数组，格式与视频帧一致
                out.write(frame)
                total_frames_written += 1

            out.release()

            if total_frames_written == 0:
                logging.warning("未能写入任何帧，请检查时间区间和视频段是否匹配。")
                return False

            logging.info(f"成功生成合并视频：{output_file}，共写入{total_frames_written}帧。")
            return output_file

        except Exception as e:
            logging.error(f"合并视频时发生错误: {e}")
            return False

    def euclidean_distance(self, point1, point2):
        """
        计算两点之间的欧氏距离。

        :param point1: tuple, (x1, y1)
        :param point2: tuple, (x2, y2)
        :return: float
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_mouth_opening(self, landmarks, image_width, image_height):
        """
        计算嘴巴开合度，通过上下嘴唇中心点的垂直距离。

        :param landmarks: list of landmarks
        :param image_width: int
        :param image_height: int
        :return: float
        """
        # MediaPipe Face Mesh 中嘴部关键点索引
        # 上嘴唇中心点和下嘴唇中心点
        # 13: 上嘴唇中心, 14: 下嘴唇中心
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]

        top_lip_y = top_lip.y * image_height
        bottom_lip_y = bottom_lip.y * image_height
        opening = abs(bottom_lip_y - top_lip_y)
        return opening

    def assign_face_ids(self, current_faces, previous_faces, threshold=50):
        """
        为当前帧的人脸分配唯一的face_id，基于与上一帧的人脸位置的距离。

        :param current_faces: list of tuples, [(x, y), ...]
        :param previous_faces: dict, {face_id: (x, y), ...}
        :param threshold: float, 最大匹配距离
        :return: list of face_ids
        """
        face_ids = []
        updated_previous_faces = previous_faces.copy()

        for face in current_faces:
            x, y = face
            min_distance = float('inf')
            matched_id = None
            for face_id, pos in previous_faces.items():
                distance = self.euclidean_distance((x, y), pos)
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    matched_id = face_id
            if matched_id is not None:
                face_ids.append(matched_id)
                updated_previous_faces[matched_id] = (x, y)  # 更新位置
            else:
                new_id = max(updated_previous_faces.keys(), default=0) + 1
                face_ids.append(new_id)
                updated_previous_faces[new_id] = (x, y)
        return face_ids, updated_previous_faces

    def process_video(self, video_path):
        """
        处理视频，检测说话者并记录其所有说话帧的人脸位置。

        :param video_path: str，视频文件路径
        :return: (most_talking_face_id, face_positions)
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        # 存储每个face_id的说话帧数和人脸位置
        talking_frames = defaultdict(list)  # {face_id: [frame_numbers]}
        face_positions = defaultdict(list)   # {face_id: [(x, y, w, h), ...]}

        # 用于跟踪人脸ID
        previous_faces = {}  # {face_id: (x, y)}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image_height, image_width, _ = frame.shape

            # 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 处理帧
            results = self.face_mesh.process(frame_rgb)

            current_faces = []
            current_landmarks = []
            bounding_boxes = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 计算嘴巴开合度
                    mouth_opening = self.calculate_mouth_opening(face_landmarks.landmark, image_width, image_height)

                    # 获取人脸边界框（使用关键点计算边界）
                    x_coords = [lm.x for lm in face_landmarks.landmark]
                    y_coords = [lm.y for lm in face_landmarks.landmark]
                    x_min = int(min(x_coords) * image_width)
                    x_max = int(max(x_coords) * image_width)
                    y_min = int(min(y_coords) * image_height)
                    y_max = int(max(y_coords) * image_height)
                    bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)
                    bounding_boxes.append(bounding_box)

                    # 中心点
                    center_x = x_min + (x_max - x_min) / 2
                    center_y = y_min + (y_max - y_min) / 2
                    current_faces.append((center_x, center_y))
                    current_landmarks.append(mouth_opening)

            # 分配face_id
            face_ids, previous_faces = self.assign_face_ids(current_faces, previous_faces, self.face_tracking_threshold)

            # 判断说话并记录
            for idx, face_id in enumerate(face_ids):
                mouth_opening = current_landmarks[idx]
                if mouth_opening > self.mouth_open_threshold:
                    talking_frames[face_id].append(frame_count)
                    face_positions[face_id].append((
                        frame_count,  # frame_number
                        bounding_boxes[idx][0],  # x
                        bounding_boxes[idx][1],  # y
                        bounding_boxes[idx][2],  # w
                        bounding_boxes[idx][3]  # h
                    ))

            # 可视化调试

            # if results.multi_face_landmarks:
            #     for face_id, bbox in zip(face_ids, bounding_boxes):
            #         x, y, w, h = bbox
            #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #         cv2.putText(frame, f'ID: {face_id}', (x, y - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


        cap.release()
        # cv2.destroyAllWindows()

        if not talking_frames:
            print("视频中没有检测到说话的人。")
            return None

        # 找到说话帧数最多的face_id
        most_talking_face_id = max(talking_frames, key=lambda k: len(talking_frames[k]))
        print(f"说话帧数最多的 face_id: {most_talking_face_id}, 说话帧数: {len(talking_frames[most_talking_face_id])}")

        # 返回该face_id的所有人脸位置
        return face_positions[most_talking_face_id]

    def recognize_identity(self, video_path, face_positions):
        """
        识别说话者的身份，返回平均特征向量。

        :param video_path: str, 视频文件路径
        :param face_positions: list of tuples, [(frame_number, x, y, w, h), ...]
        :return: np.ndarray, 平均特征向量
        """
        cap = cv2.VideoCapture(video_path)
        speaker_features = []

        for pos in face_positions:
            frame_number, x, y, w, h = pos
            # 设置视频到特定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            if not ret:
                continue

            # 裁剪人脸区域
            face_image = frame[y:y + h, x:x + w]
            if face_image.size == 0:
                continue

            # 转换为 PIL Image
            face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

            # 对齐和裁剪
            aligned_face = get_aligned_face(image_path=None, rgb_pil_image=face_image_pil)
            if aligned_face is None:
                continue
            print("已对齐")

            # # 保存对齐后的人脸图像到临时路径
            # aligned_image_path = f"aligned_temp_{frame_number}.jpg"
            # aligned_face.save(aligned_image_path)

            # 预处理图像
            imgs = preprocess_image(aligned_face)
            # print("开始提取特征")
            # print(type(imgs))
            # print(imgs[1].shape)
            # 提取特征
            feature = extract_feature(imgs, self.rknn)
            print('特征已经提取')

            speaker_features.append(feature)

        cap.release()
        self.rknn.release()

        if not speaker_features:
            print("未能获取说话者的人脸编码。")
            return None

        # 计算所有特征的平均值
        avg_feature = np.mean(speaker_features, axis=0)
        print('avg_feature')

        return avg_feature

    def get_most_talking_person(self, video_path):
        """
        获取视频中说话帧数最多的人的平均特征向量。

        :param video_path: str, 视频文件路径
        :return: dict, 包含face_id, average_feature
        """
        talking_face_positions = self.process_video(video_path)

        if talking_face_positions is not None:
            print(f"说话帧数: {len(talking_face_positions)}")

            # 提取说话者的平均特征向量
            average_feature = self.recognize_identity(video_path, talking_face_positions)
            if average_feature is not None:
                print(f"说话者的平均特征向量已提取。")
            else:
                print(f"未能提取说话者的平均特征向量。")

            # 返回结果
            # print(average_feature)
            return average_feature

        else:
            print("视频中没有检测到说话的人。")
            return None


# if __name__=="__main__":
#
#     def compute_face_similarity(feature1, feature2):
#         print(type(feature1))
#         print(type(feature2))
#         feature1 -= np.mean(feature1, axis=1, keepdims=True)
#         feature1 /= np.linalg.norm(feature1, axis=1, keepdims=True)
#         print(1111)
#         feature2 -= np.mean(feature2, axis=1, keepdims=True)
#         feature2 /= np.linalg.norm(feature2, axis=1, keepdims=True)
#         print(1111)
#         # Compute similarity (cosine similarity)
#         similarity = np.dot(feature1, feature2.T)
#         # Since features are 1 x N, similarity is a scalar
#         similarity = similarity[0][0]
#         return similarity
#
#     recognizer = VideoProcessorNPU()
#     video_path = r"E:\python_project\speaker\tem\videos\segment_20241225_170317_20241225_170327.mp4"  # 替换为实际视频路径
#     net = load_recognition_model(r'FaceNet\models\mobileface.ckpt')
#     # 获取说话者的平均特征
#     img = r"D:\python_project\speaker\FaceNet\data\detected_faces\132465.jpg"
#     result = recognizer.get_most_talking_person(video_path)
#     if result is not None:
#         print("result.shape", result.shape)
#         imgs = preprocess_image(img)
#         feature2 = extract_feature(imgs=imgs, net = net)
#         print("feature2.shape", feature2.shape)
#         s = compute_face_similarity(result,feature2)
#
#
#         if result is None:
#             print("未检测到任何说话者。")
#         else:
#             # print(f"说话者的平均特征向量: {result}")
#             print(s)
#
#     else: print("error 1111")
