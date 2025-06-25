import os
import sys
import cv2
import numpy as np

nPoints = 15
POSE_PAIRS = [[0, 1],  [1, 2],  [2, 3],
              [3, 4],  [1, 5],  [5, 6],
              [6, 7],  [1, 14], [14, 8],
              [8, 9],  [9, 10], [14, 11],
              [11, 12],[12, 13],]

def detect_objects(net, im):
    netInputSize = (368, 368)
    blob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    return output

# Hiển thị đối tượng
def display_objects(im, output, threshold=0.1):
    inWidth  = im.shape[1]
    inHeight = im.shape[0]
    scaleX = inWidth  / output.shape[3]
    scaleY = inHeight / output.shape[2]
    points = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = scaleX * point[0]
        y = scaleY * point[1]
        if prob > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(im, points[partA], points[partB], (255, 255, 0), 2)
            cv2.circle(im, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    return im

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# Thiếp lập camera
source = cv2.VideoCapture(s)
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

base_path   = os.path.dirname(os.path.abspath(__file__))
protoFile   = os.path.join(base_path, "models/pose_estimation_openpose/pose_deploy_linevec_faster_4_stages.prototxt")
weightsFile = os.path.join(base_path, "models/pose_estimation_openpose/pose_iter_160000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    output = detect_objects(net, frame)
    frame = display_objects(frame, output)
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)