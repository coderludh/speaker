from face_alignment.align import get_aligned_face
from PIL import Image
import os

import os
from PIL import Image


def align_and_save_images(input_folder, output_folder):
    # 创建输出文件夹，如果不存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 确保是图片文件（例如 .jpg, .png）
        if file_path.endswith(('.jpg', '.jpeg', '.png')):
            print(f"正在处理：{filename}")

            # 获取对齐后的图像
            aligned_face = get_aligned_face(file_path)

            if aligned_face is not None:
                # 保存到输出文件夹，文件名保持不变
                output_path = os.path.join(output_folder, filename)
                aligned_face.save(output_path)
                print(f"已保存对齐人脸图像: {output_path}")
            else:
                print(f"未检测到人脸，跳过：{filename}")
        else:
            print(f"跳过非图像文件: {filename}")


if __name__ == '__main__':
    input_folder = r"D:\python_project\FaceNet\data\test"
    output_folder = r"D:\python_project\FaceNet\data\detected_faces"
    align_and_save_images(input_folder, output_folder)





