# 本地请求
import time

import requests
start_time = time.time()
# 设置 URL 和文件路径
url = 'http://127.0.0.1:8000/upload_video'
file_path = r".\tem\videos\1.mp4"

# 打开文件并发送请求
with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# 输出响应状态码和内容
print(f"Status Code: {response.status_code}")
print("Response Text:")
print(response.json())
print(f"process time:{time.time() - start_time} Seconds")

#  时间戳请求
# import requests
#
# url = 'http://127.0.0.1:8000/process_face_video'
# data = {
#     'start_time': 1735133326.81693,
#     'end_time': 1735133367.1943593,
# }
#
# response = requests.post(url, data=data)
# print(response.json())
