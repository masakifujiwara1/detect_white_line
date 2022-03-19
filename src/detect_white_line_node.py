import std_msgs
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# matplotlib inline

path = "/home/fmasa/catkin_ws/src/detect_white_line/image/20220314_15"

img = cv2.imread(path + "2144.jpg")

# plt.figure(figsize=(5, 5))
#img2 = img[:, :, ::-1]
# plt.xticks([]), plt.yticks([])
# plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(gray)
plt.gray()

edges = cv2.Canny(gray, 100, 200)


plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(edges)

lines = cv2.HoughLinesP(edges,
                        rho=1,
                        theta=np.pi/360,
                        threshold=100,
                        minLineLength=100,
                        maxLineGap=10)
# print(lines[:5])

for line in lines:
    x1, y1, x2, y2 = line[0]

    # 赤線を引く
    red_line_img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

img2 = img[:, :, ::-1]

plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(img2)

plt.show()
