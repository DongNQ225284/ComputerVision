'''
Mini project:
    Nhận diện vật thể trong thời gian thực

Mô tả:
    Sử dụng camera phát hiện vật thể và gán nhãn trong thời gian thực
    Mini project sử dụng Tensorflow với thư viện ssd_mobilenet_v2_coco
'''

import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))
classFile = os.path.join(base_path, "models/ssd_mobilenet_v2_coco/coco_class_labels.txt")
# Đọc mô hình Tensorflow
modelFile  = os.path.join(base_path, "models/ssd_mobilenet_v2_coco/frozen_inference_graph.pb")
configFile = os.path.join(base_path, "models/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

# Kiểm tra tính hợp lệ
with open(classFile) as fp:
    labels = fp.read().split("\n")
# with open(modelFile) as fp:
#     labels = fp.read().split("\n")
# with open(configFile) as fp:
#     labels = fp.read().split("\n")


# Đối với mỗi tệp trong thư mục
def detect_objects(net, im, dim = 300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    return objects

# Hiển thị văn bản
def display_text(im, text, x, y):
    FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    THICKNESS = 1
    # Nhận kích thước văn bản
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]
    # Sử dụng kích thước văn bản để tạo hình chữ nhật màu đen
    cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0),cv2.FILLED)
    cv2.putText(im, text, (x, y - 5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)

# Hiển thị đối tượng
def display_objects(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    #mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# Thiếp lập camera
source = cv2.VideoCapture(s)
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Đọc mạng Tensorflow
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    objects = detect_objects(net, frame)
    frame = display_objects(frame, objects)
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)