import time

import requests
start_time = time.time()
# 设置 URL 和文件路径
url = 'http://127.0.0.1:8000/upload_audio'
file_path = r"./data/sound/D4_766.wav"

# 打开文件并发送请求
with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# 输出响应状态码和内容
print(f"Status Code: {response.status_code}")
print("Response Text:")
print(response.json())
print(f"process time:{time.time() - start_time} Seconds")