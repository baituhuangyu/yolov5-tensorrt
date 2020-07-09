# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import cv2  # OpenCV
# import tkinter.filedialog  # Python文件对话框
#
# filename = tkinter.filedialog.askopenfilename()  # 弹出对话框选择需要裁剪的视频文件

filename = "/mnt/data/yu.huang/share_data/CUHKSquare.mpg"
cap = cv2.VideoCapture(filename)  # 打开视频文件
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获得视频文件的帧数
fps = cap.get(cv2.CAP_PROP_FPS)  # 获得视频文件的帧率
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获得视频文件的帧宽
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获得视频文件的帧高

# 创建保存视频文件类对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (int(width), int(height)))

# 计算视频长度/s
video_length = frames / fps
print('start and stop must < %.1f' % video_length)  # 提示用户输入变量的范围
start = float(input('Input an start time/s:'))
stop = float(input('Input an stop time/s:'))
# 设置帧读取的开始位置
cap.set(cv2.CAP_PROP_POS_FRAMES, start * fps)
pos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # 获得帧位置
while (pos <= stop * fps):
    ret, frame = cap.read()  # 捕获一帧图像
    out.write(frame)  # 保存帧
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

cap.release()
out.release()

