# import cv2
# import mediapipe as mp
# import numpy as np
#
# # 初始化 MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
# # 初始化摄像头
# cap = cv2.VideoCapture(0)
#
# # 定义嘴唇关键点索引
# LIPS = [
#     61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
#     185, 40, 39, 37, 0, 267, 269, 270, 409,
#     78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
#     191, 80, 81, 82, 13, 312, 311, 310, 415
# ]
#
# # 去重处理
# LIPS = list(set(LIPS))
#
# # 初始化变量
# prev_lip_landmarks = None
# speaking = False
# threshold = 2.0  # 根据实验调整阈值
# speak_frames = 0
# required_frames = 5  # 连续帧数超过此值判定为说话
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(frame_rgb)
#
#     h, w, _ = frame.shape
#
#     if results.multi_face_landmarks:
#         face_landmarks = results.multi_face_landmarks[0]
#
#         # 获取当前帧的嘴唇关键点位置
#         lip_landmarks = []
#         for idx in LIPS:
#             x = int(face_landmarks.landmark[idx].x * w)
#             y = int(face_landmarks.landmark[idx].y * h)
#             lip_landmarks.append((x, y))
#
#         # 可视化嘴唇关键点
#         for (x, y) in lip_landmarks:
#             cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#
#         # 与前一帧比较
#         if prev_lip_landmarks is not None:
#             # 计算当前帧与前一帧的关键点移动距离
#             movement = np.mean([np.linalg.norm(np.array(curr) - np.array(prev))
#                                 for curr, prev in zip(lip_landmarks, prev_lip_landmarks)])
#
#             if movement > threshold:
#                 speak_frames += 1
#                 if speak_frames >= required_frames:
#                     speaking = True
#             else:
#                 speak_frames = 0
#                 speaking = False
#         else:
#             movement = 0.0
#
#         prev_lip_landmarks = lip_landmarks
#
#         # 显示说话状态
#         if speaking:
#             cv2.putText(frame, "Speaking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2, cv2.LINE_AA)
#         else:
#             cv2.putText(frame, "Not Speaking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 255, 0), 2, cv2.LINE_AA)
#
#     else:
#         prev_lip_landmarks = None
#         speaking = False
#         cv2.putText(frame, "Face Not Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (0, 255, 255), 2, cv2.LINE_AA)
#
#     # 显示视频流
#     cv2.imshow('Speaking Detection', frame)
#
#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
