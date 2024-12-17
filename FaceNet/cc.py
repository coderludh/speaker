# main.py

from face_database import FaceDatabase
from camera import Camera

def main():
    # 设置图片库文件夹和模型路径
    image_folder = r'.\data\test'  # 替换为您的图片库路径
    model_path = r'.\models\mobileface.ckpt'  # 替换为您的模型文件路径

    # 构建人脸数据库
    print("构建人脸数据库中...")
    face_db = FaceDatabase(image_folder=image_folder, model_path=model_path, device='cuda:0')
    database = face_db.get_database()
    net = face_db.get_model()
    print("人脸数据库构建完成。")

    # 运行摄像头识别
    webcam = Camera(face_database=database, net=net, device='cuda:0', threshold=0.5)
    webcam.run()

if __name__ == '__main__':
    main()
