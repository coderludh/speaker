# face_database.py

import os
import torch
import numpy as np
from PIL import Image
from face_alignment.mtcnn import MTCNN
from models.mobileface import MobileFacenet


class FaceDatabase:
    def __init__(self, image_folder, model_path, device='cuda:0'):
        self.image_folder = image_folder
        self.model_path = model_path
        self.device = device
        self.mtcnn = MTCNN(device=self.device, crop_size=(96, 112))
        self.net = self.load_recognition_model()
        self.database = self.build_database()

    def load_recognition_model(self):
        """
        加载人脸识别模型。
        """
        net = MobileFacenet()
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            net = net.cuda()
        ckpt = torch.load(self.model_path, map_location=self.device)
        net.load_state_dict(ckpt['net_state_dict'])
        net.eval()
        return net

    def preprocess_image(self, pil_image):
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

    def build_database(self):
        """
        遍历图片文件夹，检测和对齐人脸，提取特征并构建数据库。
        """
        face_database = {}
        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(self.image_folder, filename)
                name = os.path.splitext(filename)[0]  # 文件名作为名字

                # 加载图像
                img = Image.open(image_path).convert('RGB')

                # 人脸检测和对齐
                try:
                    boxes, faces = self.mtcnn.align_multi(img, limit=1)
                    if len(faces) == 0:
                        print(f"未检测到人脸: {image_path}")
                        continue
                    aligned_face = faces[0]
                except Exception as e:
                    print(f"人脸检测失败: {image_path}")
                    print(e)
                    continue

                # 预处理并提取特征
                imgs = self.preprocess_image(aligned_face)
                feature = self.extract_feature(imgs)
                # 归一化特征
                feature -= np.mean(feature, axis=1, keepdims=True)
                feature /= np.linalg.norm(feature, axis=1, keepdims=True)

                face_database[name] = feature
                print(f"已添加 {name} 到人脸数据库。")
        return face_database

    def get_database(self):
        return self.database

    def get_model(self):
        return self.net
