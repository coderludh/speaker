from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# 初始化 MTCNN
mtcnn = MTCNN(keep_all=True, device='cpu')  # 如果有 GPU，可以将 device 设置为 'cuda'

# 加载图像
image_path = r"D:\python_project\FaceNet\data\lu2.jpg"
image = Image.open(image_path)

# 检测人脸
boxes, probs = mtcnn.detect(image)

# 打印检测结果
print("Detected boxes:", boxes)
print("Detection probabilities:", probs)

# 创建保存人脸图像的目录
output_dir = r"D:\python_project\FaceNet\data\detected_faces"
os.makedirs(output_dir, exist_ok=True)


# 可视化检测结果并保存裁剪的人脸
def draw_boxes_and_save_faces(image, boxes, probs, output_dir, confidence_threshold=0.90, desired_size=(96, 112)):
    """
    在图像上绘制人脸边界框，保存裁剪的人脸图像，并将裁剪后的图像调整为指定尺寸。

    参数:
        image (PIL.Image): 原始图像。
        boxes (ndarray): 人脸边界框坐标。
        probs (ndarray): 每个边界框的置信度。
        output_dir (str): 保存裁剪人脸图像的目录。
        confidence_threshold (float): 置信度阈值，低于此值的检测结果将被忽略。
        desired_size (tuple): 裁剪后图像的目标尺寸 (宽度, 高度)。

    返回:
        PIL.Image: 绘制了边界框的图像。
    """
    draw = ImageDraw.Draw(image)
    face_count = 2  # 初始化人脸计数器，从1开始
    if boxes is not None:
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < confidence_threshold:
                continue
            # 绘制边界框
            draw.rectangle(box.tolist(), outline=(0, 255, 0), width=2)
            # 在边界框上方添加置信度
            draw.text((box[0], box[1] - 10), f"{prob:.2f}", fill=(255, 0, 0))

            # 裁剪人脸区域
            x1, y1, x2, y2 = map(int, box)
            face = image.crop((x1, y1, x2, y2))

            # 调整裁剪后的人脸图像大小
            face = face.resize(desired_size, Image.BILINEAR)

            # 保存裁剪并调整大小的人脸
            face_filename = os.path.join(output_dir, f"face_{face_count}.jpg")
            face.save(face_filename)
            print(f"Saved face {face_count} to {face_filename}")
            face_count += 1
    return image


# 绘制边界框并保存人脸
image_with_boxes = draw_boxes_and_save_faces(image.copy(), boxes, probs, output_dir)

# 使用 matplotlib 显示图像
plt.figure(figsize=(10, 10))
plt.imshow(image_with_boxes)
plt.axis('off')  # 关闭坐标轴
plt.show()

# 或者保存可视化结果
# image_with_boxes.save(r"D:\python_project\FaceNet\data\zhou_with_boxes.jpg")
