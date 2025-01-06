import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from FaceNet.face_alignment import mtcnn  # 使用 MTCNN 进行人脸检测
from FaceNet.models.mobileface import MobileFacenet  # 你的识别模型

# 初始化 MTCNN 人脸检测模型
mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))


# 加载识别模型
def load_recognition_model(model_path):
    net = MobileFacenet()

    net = net.cpu()
    # 加载模型权重
    ckpt = torch.load(model_path, map_location='cpu')
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    return net


# 图像预处理
def preprocess_image(img):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.resize((112, 112))
    img = np.array(img)
    img = img.astype(np.float32)

    if len(img.shape) == 2:  # If grayscale, convert to RGB
        img = np.stack([img]*3, axis=2)
    # Create a list with original and horizontally flipped images
    img_list = [img, img[:, ::-1, :]]
    for i in range(len(img_list)):
        # Normalize the image
        img_list[i] = (img_list[i] - 127.5) / 128.0
        # Transpose to (channel, height, width)
        img_list[i] = img_list[i].transpose(2, 0, 1)
        img_list[i] = np.expand_dims(img_list[i], axis=0)
    return img_list


# 提取人脸特征
def extract_feature(imgs, rknn):
    features = []
    for i in range(len(imgs)):
        feature = rknn.inference(inputs=[imgs[i]], data_format='nchw')
        features.append(feature[0])
    feature = np.concatenate((features[0], features[1]), axis=1)
    return feature


# 在图像上显示人脸框和名字
def draw_face_boxes(image, boxes, recognized_names):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, box in enumerate(boxes):
        left, top, right, bottom = map(int, box[:4])
        # 绘制矩形框
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        # 显示识别结果
        name = recognized_names[idx] if recognized_names[idx] else "Unknown"
        text_size = draw.textsize(name, font=font)
        # 绘制文本背景
        draw.rectangle([left, top - text_size[1], left + text_size[0], top], fill="red")
        draw.text((left, top - text_size[1]), name, fill="white", font=font)
    return image


# 读取图片库并提取特征
def load_image_database(image_paths, model_path):
    database = {}
    net = load_recognition_model(model_path)

    for image_path in image_paths:
        imgs = preprocess_image(image_path)
        feature = extract_feature(imgs, net)
        feature -= np.mean(feature, axis=1, keepdims=True)
        feature /= np.linalg.norm(feature, axis=1, keepdims=True)
        name = image_path.split('/')[-1].split('.')[0]  # 提取文件名作为名字
        database[name] = feature
    return database, net


# 识别人脸
def recognize_face(frame, database, net, threshold=0.5):
    # 将 OpenCV 图像转换为 PIL 图像
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn_model.align_multi(pil_img)

    recognized_names = []
    for box in boxes:
        # 获取人脸区域
        left, top, right, bottom = map(int, box[:4])
        face_img = frame[top:bottom, left:right]

        # 如果人脸区域大小不符合预期，跳过
        if face_img.size == 0:
            recognized_names.append(None)
            continue

        # 预处理检测到的脸并提取特征
        imgs = preprocess_image(face_img)
        feature = extract_feature(imgs, net)
        feature -= np.mean(feature, axis=1, keepdims=True)
        feature /= np.linalg.norm(feature, axis=1, keepdims=True)

        # 与库中的每个特征进行比较
        best_match_name = None
        best_match_score = -1
        for name, db_feature in database.items():
            similarity = np.dot(db_feature, feature.T)[0][0]
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_name = name

        # 判断是否匹配
        if best_match_score > threshold:
            recognized_names.append(best_match_name)
        else:
            recognized_names.append(None)  # 未识别出此人

    # 在检测到的人脸框上绘制框和名字
    pil_img_with_boxes = draw_face_boxes(pil_img, boxes, recognized_names)
    # 将 PIL 图像转换回 OpenCV 格式
    frame_with_boxes = cv2.cvtColor(np.array(pil_img_with_boxes), cv2.COLOR_RGB2BGR)
    return frame_with_boxes


# 人脸对齐
def get_aligned_face(image_path, rgb_pil_image=None):
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0]
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        face = None

    return face


# 主函数
def main(video_source=0, image_database_paths=[], model_path="path_to_your_model.pth"):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # 加载图像库和识别模型
    database, net = load_image_database(image_database_paths, model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 人脸识别
        frame_with_boxes = recognize_face(frame, database, net, threshold=0.5)

        # 显示图像
        cv2.imshow('Face Recognition', frame_with_boxes)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


