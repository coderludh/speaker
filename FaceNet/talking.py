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
# # 定义嘴部关键点索引
# UPPER_LIP_TOP = 13
# LOWER_LIP_BOTTOM = 14
#
# # 初始化说话检测变量
# speaking = False
# threshold = 0.02  # 根据实际情况调整
# prev_lip_distance = None
# speak_frames = 0
# required_frames = 10  # 连续帧数超过此值判定为说话
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(frame_rgb)
#
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             h, w, _ = frame.shape
#             # 获取嘴部关键点坐标
#             upper_lip = face_landmarks.landmark[UPPER_LIP_TOP]
#             lower_lip = face_landmarks.landmark[LOWER_LIP_BOTTOM]
#
#             upper_lip_y = upper_lip.y * h
#             lower_lip_y = lower_lip.y * h
#             lip_distance = lower_lip_y - upper_lip_y
#
#             # 计算嘴唇距离变化
#             if prev_lip_distance is not None:
#                 delta = abs(lip_distance - prev_lip_distance)
#                 if delta > threshold:
#                     speak_frames += 1
#                     if speak_frames > required_frames:
#                         speaking = True
#                 else:
#                     speak_frames = 0
#                     speaking = False
#             prev_lip_distance = lip_distance
#
#             # 可视化嘴部关键点
#             cv2.circle(frame, (int(upper_lip.x * w), int(upper_lip.y * h)), 2, (0, 255, 0), -1)
#             cv2.circle(frame, (int(lower_lip.x * w), int(lower_lip.y * h)), 2, (0, 255, 0), -1)
#
#             # 显示说话状态
#             if speaking:
#                 cv2.putText(frame, "Speaking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                             1, (0, 0, 255), 2, cv2.LINE_AA)
#             else:
#                 cv2.putText(frame, "Not Speaking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                             1, (0, 255, 0), 2, cv2.LINE_AA)
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
