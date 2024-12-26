# export_to_onnx.py
import torch
from torch.autograd import Variable
from FaceNet.models.mobileface import MobileFacenet

def load_recognition_model(model_path):
    net = MobileFacenet()
    net = net.cpu()
    # 加载模型权重
    ckpt = torch.load(model_path, map_location='cpu')
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    return net

if __name__ == "__main__":
    # 设置模型路径和输出路径
    checkpoint_path = 'E:\python_project\speaker\FaceNet\models\mobileface.ckpt'  # 请替换为您的 .ckpt 文件路径
    onnx_output_path = 'E:\python_project\speaker\FaceNet\models\mobilefacenet.onnx'

    # 加载模型
    model = load_recognition_model(checkpoint_path)

    # 创建示例输入
    dummy_input = torch.randn(1, 3, 112, 112)  # 根据您的模型输入尺寸调整

    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX 模型已成功导出为 {onnx_output_path}")
