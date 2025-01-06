import cv2
import numpy as np
from rknnlite.api import RKNNLite

rknn = RKNNLite(verbose=True)

# Set inputs
img = cv2.imread('./data/pic/xu.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
img = (img - 127.5 ) / 128.0
img = img.transpose(2,0,1)
img = np.expand_dims(img, 0)
print(img)
print(img.shape)
ret = rknn.load_rknn(path='./FaceNet/models/mobilefacenet.rknn')
# Init runtime environment
print('--> Init runtime environment')
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

# Inference
print('--> Running model')
outputs = rknn.inference(inputs=[img],data_format=['nchw'])
print('done')
print(outputs)
rknn.release()
