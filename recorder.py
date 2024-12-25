import cv2
import threading
import os
from sqlalchemy.engine import TupleResult
import time
import json
from collections import deque
from datetime import datetime
import time




class VideoRecorder:
    def __init__(self, output_dir='tem/videos', log_file='tem/segments_log.json', segment_duration=10, frame_width=640, frame_height=480, fps=30):
        self.output_dir = output_dir
        self.log_file = log_file
        self.segment_duration = segment_duration
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        # if os.path.exists(self.log_file):
        #     os.remove(self.log_file)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

        self.running = False

        self.current_frames = []
        self.current_segment_start = None
        self.current_segment_end = None

        # 内存缓冲区
        self.memory_buffer = deque()

        # 线程和锁,可以设置daemon为Ture，设置为守护线程，主线程结束时自动停止
        self.thread = threading.Thread(target=self.record)
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread.start()
        print('开始录制视频')

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()
        cv2.destroyAllWindows()
        print('停止录制视频')


    def record(self):
        segment_start_time = time.time()
        self.current_segment_start = time.time()

        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                print("无法读取视屏")
                continue

            #获取当前时间戳
            current_time = time.time()

            # # 在帧上叠加视屏戳
            # cv2.putText(frame, current_time, (10, self.frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

            with self.lock:
                self.current_frames.append(frame.copy())

            # 在内存添加当前帧，以及时间戳
            # 锁的作用是防止共享数据在别的地方被修改
            with self.lock:
                self.memory_buffer.append((current_time, frame.copy()))

            # 判断是否录制够了当前时长的视屏
            elapsed_time = time.time() - segment_start_time
            if elapsed_time > self.segment_duration:
                self.current_segment_end = time.time()
                self.save_segment_with_logging()
                with self.lock:
                    self.current_frames = []
                segment_start_time = time.time()
                self.current_segment_start = time.time()
        # 保存最后一段
        if self.current_frames:
            self.current_segment_end = time.time()
            self.save_segment_with_logging()

    def save_segment_with_logging(self):
        if not self.current_frames:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        start_str = datetime.fromtimestamp(self.current_segment_start).strftime("%Y%m%d_%H%M%S")
        end_str = datetime.fromtimestamp(self.current_segment_end).strftime("%Y%m%d_%H%M%S")
        filename = f"segment_{start_str}_{end_str}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.frame_width, self.frame_height))

        for frame in self.current_frames:
            out.write(frame)

        out.release()
        print(f"保存视屏{filename}")

        # 记录日志
        log_entry = {
            "filename": filename,
            # 转化为iso格式，格式为"2023-12-12T14:30:45+00:00"
            "start_time": self.current_segment_start,
            "end_time": self.current_segment_end
        }

        self.write_log(log_entry)

    def write_log(self, entry):
        with open(self.log_file, 'r+') as f:
            data = json.load(f)
            data.append(entry)
            # 将文件指针移动到文件开头，重写文件内容
            f.seek(0)
            json.dump(data, f, indent=4)
        print(f"记录日志{entry}")

    # 得到指定时间之前的视屏名称
    def get_videos_before(self, timestamp):
        try:
            target_time = timestamp
        except ValueError as e:
            print("删除视屏传入无效时间戳")
            return []

        with open(self.log_file, 'r') as f:
            data = json.load(f)

        before_videos = []
        for entry in data:
            end_time = entry['end_time']
            if end_time <= target_time:
                before_videos.append(entry['filename'])

        return before_videos

    def delete_videos_before(self, timestamp):
        """
        删除给定时间戳之前的视屏和日志
        :param timestamp:
        :return:
        """

        before_videos = self.get_videos_before(timestamp)
        if not before_videos:
            print(f"没有在{timestamp}前的视频")
            return

        for filename in before_videos:
            file_path = os.path.join(self.output_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"删除文件发生错误{e}")
            else:
                print(f"文件不存在: {filename}")

        with open(self.log_file, 'r+') as f:
            data = json.load(f)
            data = [entry for entry in data if entry['filename'] not in before_videos]
            f.seek(0)
            json.dump(data, f, indent=4)
            # 截断到当前文件指针的位置，确保日志只有data中的数据
            f.truncate()
        print("已从日志删除视屏")

    # 获取内存中的帧
    def get_recent_memory_buffer(self, cutoff_time):
        recent_frames = []

        with self.lock:
            for timestamp, frame in self.memory_buffer:
                if timestamp <= cutoff_time:
                    recent_frames.append(frame)
                else:
                    break  # 由于是按时间顺序排列，可以提前退出
            recent_frames.reverse()  # 还原帧的顺序

        return recent_frames

    def delete_all_videos(self):
        """
        删除所有已保存的视频段及其日志记录。
        """
        with self.lock:
            # 获取所有视频文件名
            videos_to_delete = os.listdir(self.output_dir)

            # 删除视频文件
            for filename in videos_to_delete:
                filepath = os.path.join(self.output_dir, filename)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        print(f"已删除视频段: {filename}")
                    except Exception as e:
                        print(f"删除视频 {filename} 时发生错误: {e}")
                else:
                    print(f"文件不存在: {filename}")

            # 清空日志文件
            with open(self.log_file, 'w') as f:
                json.dump([], f)
            print("所有视频段已从日志中移除。")







if __name__ == "__main__":
    segmenter = VideoRecorder(
        output_dir='tem/videos',
        log_file='tem/segments_log.json',
        segment_duration=10,  # 每段10秒
        frame_width=640,
        frame_height=480,
        fps=30
    )
    segmenter.delete_all_videos()
    try:
        segmenter.start()
        print("按下 'q' 键停止录制。")
        # 主线程中的循环
        # while segmenter.running:
        #     time.sleep(1)
    except KeyboardInterrupt:
        segmenter.stop()

























