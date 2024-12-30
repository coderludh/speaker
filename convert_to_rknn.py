# convert_to_rknn.py
from rknn.api import RKNN

# 初始化 RKNN 对象
rknn = RKNN()

# 配置 RKNN
print('--> 配置 RKNN')
rknn.config(
    reorder_channel='0 1 2',              # 通道顺序，默认 RGB
    target_platform='rk3588',             # 目标平台
    quantized_dtype='asymmetric_quant'     # 量化类型，可选 'dynamic_range_quant' 或 'asymmetric_quant'
)

# 加载 ONNX 模型
print('--> 加载 ONNX 模型')
ret = rknn.load_onnx(model='mobilefacenet.onnx')
if ret != 0:
    print('Load ONNX model failed!')
    exit(ret)

# 编译 RKNN 模型，启用量化
print('--> 编译 RKNN 模型')
ret = rknn.build(
    do_quantization=True,
    dataset='dataset.txt'
)
if ret != 0:
    print('Build RKNN model failed!')
    exit(ret)

# 导出 RKNN 模型
print('--> 导出 RKNN 模型')
ret = rknn.export_rknn('mobilefacenet.rknn')
if ret != 0:
    print('Export RKNN model failed!')
    exit(ret)

# 释放 RKNN 资源
rknn.release()
print('转换为 RKNN 模型已完成！')
