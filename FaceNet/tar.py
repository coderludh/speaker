import tarfile

# 打开 .tar 文件
with tarfile.open('models/mobilefacenet_model_best.pth.tar', 'r') as tar:
    # 解压特定文件到指定路径
    tar.extract('specific_file.txt', path='models')