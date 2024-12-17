# webcam.py

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from face_alignment.mtcnn import MTCNN  # 导入您的 MTCNN 模型

class Camera:
    def __init__(self, face_database, net, device='cuda:0', threshold=0.5):
        self.face_database = face_database
        self.net = net
        self.device = device
        self.threshold = threshold
        self.mtcnn = MTCNN(device=self.device, crop_size=(112, 112))
        self.font = ImageFont.load_default()

    def preprocess_face(self, pil_image):
        """
        预处理 PIL 图像，准备输入到识别模型。
        """
        img = pil_image.resize((112, 112))
        img = np.array(img)

        if len(img.shape) == 2:  # 如果是灰度图，转换为 RGB
            img = np.stack([img] * 3, axis=2)
        img_list = [img, img[:, ::-1, :]]  # 原图和水平翻转的图像

        for i in range(len(img_list)):
            img_list[i] = (img_list[i] - 127.5) / 128.0
            img_list[i] = img_list[i].transpose(2, 0, 1)  # 转换为 (channel, height, width)
        imgs = [torch.from_numpy(i).float().unsqueeze(0) for i in img_list]
        return imgs

    def extract_feature(self, imgs):
        """
        从预处理后的图像中提取特征。
        """
        with torch.no_grad():
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                imgs = [img.cuda() for img in imgs]
            features = [self.net(img).cpu().numpy() for img in imgs]
            feature = np.concatenate((features[0], features[1]), axis=1)
        return feature

    def recognize_faces_in_frame(self, frame):
        """
        在给定的帧中检测和识别人脸，返回带有绘制人脸框和名字的图像。
        """
        # 将 OpenCV 图像转换为 PIL 图像
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 人脸检测
        try:
            boxes, faces = self.mtcnn.align_multi(pil_img)
            if len(faces) == 0:
                return frame  # 没有检测到人脸，返回原始帧
        except Exception as e:
            print("人脸检测失败:", e)
            return frame

        draw = ImageDraw.Draw(pil_img)

        for box, face in zip(boxes, faces):
            left, top, right, bottom = map(int, box[:4])

            # 预处理并提取特征
            imgs = self.preprocess_face(face)
            feature = self.extract_feature(imgs)
            # 归一化特征
            feature -= np.mean(feature, axis=1, keepdims=True)
            feature /= np.linalg.norm(feature, axis=1, keepdims=True)

            # 在数据库中进行匹配
            best_match = None
            best_score = self.threshold
            for name, db_feature in self.face_database.items():
                similarity = np.dot(feature, db_feature.T)[0][0]
                print(f"Comparing with {name}, similarity: {similarity}")
                if similarity > best_score:
                    best_score = similarity
                    best_match = name

            # 绘制人脸框
            draw.rectangle([left, top, right, bottom], outline='green', width=5)

            # 显示识别结果
            display_name = best_match if best_match is not None else 'Unknown'
            print(f"Display name: {display_name}")

            # 获取文本大小，使用 getbbox 方法
            text_bbox = self.font.getbbox(display_name)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 绘制文本背景
            draw.rectangle([left, top - text_height, left + text_width, top], width=10, fill='white')
            # 绘制文本
            draw.text((left, top - text_height), display_name, fill='black', width=10, font=self.font)

        # 将 PIL 图像转换回 OpenCV 图像
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame

    def run(self):
        """
        打开摄像头，实时检测和识别人脸。
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: 无法打开摄像头。")
            return

        print("开始实时人脸识别。按 'q' 键退出。")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: 无法获取摄像头帧。")
                break

            # 识别当前帧中的人脸
            frame = self.recognize_faces_in_frame(frame)

            # 显示帧
            cv2.imshow('人脸识别', frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
