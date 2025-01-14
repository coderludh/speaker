import cv2

def is_camera_available(max_tested=5):
    """
    检测系统中是否存在摄像头。

    参数:
    max_tested (int): 测试的摄像头索引范围，从 0 到 max_tested-1

    返回:
    bool: 如果至少有一个摄像头可用，返回 True，否则返回 False
    """
    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            cap.release()
            print(f"检测到摄像头: /dev/video{index}")
            return True
    return False

if __name__ == "__main__":
    if is_camera_available():
        print("系统中存在摄像头。")
    else:
        print("系统中未检测到摄像头。")
