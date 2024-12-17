import torch
from PIL import Image
import numpy as np
from models.mobileface import MobileFacenet  # 确保该模块路径正确
from facenet_pytorch import MTCNN

def detect_align_resize(image_path, image_size=112, margin=0):
    # 初始化 MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=image_size, margin=margin, device=device, post_process=False)

    # 加载图像
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法打开图像 {image_path}: {e}")
        return None

    # 检测并对齐人脸
    try:
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
        if boxes is None or len(boxes) == 0:
            print(f"未检测到人脸: {image_path}")
            return None

        # 提取对齐后的人脸
        aligned_faces = []
        for i, box in enumerate(boxes):
            box = [float(b) for b in box]  # 确保 box 是浮点数列表
            face = mtcnn.extract(img, [box], None)[0]  # 返回的是列表，需要取第一个元素
            aligned_faces.append(face)
        return aligned_faces
    except Exception as e:
        print(f"人脸检测/对齐过程中出现错误: {e}")
        return None

def preprocess_image(aligned_face):
    # 将图像转换为 NumPy 数组
    img = np.array(aligned_face)

    # 检查图像形状
    if len(img.shape) == 2:  # 如果是灰度图像，转换为 RGB
        img = np.stack([img] * 3, axis=2)

    # 创建原始图像和水平翻转图像的列表
    img_list = [img, np.fliplr(img)]

    for i in range(len(img_list)):
        # 归一化到 [-1, 1]
        img_list[i] = (img_list[i] - 127.5) / 128.0
        # 转置为 (C, H, W)
        img_list[i] = img_list[i].transpose(2, 0, 1)

    # 转换为 PyTorch 张量并添加批次维度
    imgs = [torch.from_numpy(i).float().unsqueeze(0) for i in img_list]
    return imgs

def extract_feature(imgs, net, device):
    net.eval()
    features = []
    with torch.no_grad():
        for img in imgs:
            img = img.to(device)
            feat = net(img)
            features.append(feat.cpu().numpy())
    # 拼接原始图像和翻转图像的特征
    feature = np.concatenate((features[0], features[1]), axis=1)
    return feature

def compare_images(image_path1, image_path2, model_path, threshold=0.5):
    # 调用人脸检测和对齐函数
    aligned_faces1 = detect_align_resize(image_path1)
    aligned_faces2 = detect_align_resize(image_path2)

    # 检查返回值
    if aligned_faces1 is None or len(aligned_faces1) == 0:
        print(f"Face not detected or alignment failed for image: {image_path1}")
        return None, None
    if aligned_faces2 is None or len(aligned_faces2) == 0:
        print(f"Face not detected or alignment failed for image: {image_path2}")
        return None, None

    # 选择第一张人脸进行比较
    aligned_face1 = aligned_faces1[0]
    aligned_face2 = aligned_faces2[0]

    # 预处理图像
    imgs1 = preprocess_image(aligned_face1)
    imgs2 = preprocess_image(aligned_face2)

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MobileFacenet()
    net.load_state_dict(torch.load(model_path, map_location=device)['net_state_dict'])
    net.to(device)
    net.eval()

    # 提取特征
    feature1 = extract_feature(imgs1, net, device)
    feature2 = extract_feature(imgs2, net, device)

    # 归一化特征
    feature1 -= np.mean(feature1, axis=1, keepdims=True)
    feature1 /= np.linalg.norm(feature1, axis=1, keepdims=True)
    feature2 -= np.mean(feature2, axis=1, keepdims=True)
    feature2 /= np.linalg.norm(feature2, axis=1, keepdims=True)

    # 计算余弦相似度
    similarity = np.dot(feature1, feature2.T)
    similarity = similarity[0][0]

    # 根据阈值判断
    if similarity > threshold:
        result = "Same person"
    else:
        result = "Different persons"

    print(f"Result: {result}")
    print(f"Similarity score: {similarity:.4f}")
    return result, similarity

if __name__ == '__main__':
    # 图像路径
    image_path1 = r'D:\python_project\FaceNet\data\xu.jpg'
    image_path2 = r'D:\python_project\FaceNet\data\zhou.jpg'
    model_path = 'D:\python_project\FaceNet\models\mobileface.ckpt'

    # 比较图像
    compare_images(image_path1, image_path2, model_path)
