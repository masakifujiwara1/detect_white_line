# from distutils.log import info
# from tkinter import image_names
# from types import GeneratorType

from multiprocessing.context import assert_spawning
from turtle import shape
from attr import assoc
from cv2 import cvtColor, threshold
import std_msgs
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy.linalg as LA
# matplotlib inline

path = "/home/fmasa/catkin_ws/src/detect_white_line/image/tsukuba/"
dir = "/home/fmasa/catkin_ws/src/detect_white_line/image/"

img = cv2.imread(path + "frame_235.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
np_img = np.array(img)

# plt.figure(figsize=(5, 5))
# img2 = img[:, :, ::-1]
# plt.xticks([]), plt.yticks([])
# plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("image/gray.jpg", gray)
img_p = cv2.imread("image/gray.jpg")

plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(img_p)

# edges = cv2.Canny(gray, 50, 150)
# cv2.imwrite("image/edges.jpg", edges)
# img_pp = cv2.imread("image/edges.jpg")

kernel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gray_y = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
gray_x = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

edges = np.sqrt(gray_x**2 + gray_y**2)
img_pp = edges

plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(img_pp)

# template matching


def template(dir_path):
    template = cv2.imread(dir_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


# template(dir + "template001.png")
# template(dir + "template002.png")
def image_vector():
    res = np.zeros([3])
    info = (str(np_img.shape)).replace('(', '').replace(')', '').split(',')
    # print(info[0], info[1], info[2])
    m = int(info[0])
    n = int(info[1])
    channel = int(info[2])

    p = np.zeros((m, n, channel))
    # print(p[0][0])
    for i in range(m):
        for j in range(n):
            for k in range(channel):
                p[i][j][k] = np_img[i][j][k]
    # print(p[0])

    num_pixel = m * n
    mup = np.sum(p, axis=0)
    mup = np.sum(mup, axis=0)
    mup = (1/(num_pixel-1)) * mup
    print(mup)

    for i in range(m):
        for j in range(n):
            ass = p[i][j] - mup
            # print(ass)
            ass_T = ass.T
            # print(ass, ass_T)
            ass_res = ass * ass_T
            ass_res = (1/(num_pixel-1)) * ass_res
            # print(ass_res)
            res += ass_res
            # print(ass_res)
    print(res)

    Rp_val, Rp_vec = LA.eig(res)
    print(Rp_val)
    print(Rp_vec)

    #                         rho=1,
    #                         theta=np.pi/360,
    #                         threshold=60,
    #                         minLineLength=200,
    #                         maxLineGap=10)
    # print(lines[:5])


    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     # 赤線を引く
    #     red_line_img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 30)
    # cv2.imwrite("image/output.jpg", red_line_img)
    # img_ppp = cv2.imread('image/output.jpg')
    # img2 = img[:, :, ::-1]
plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(img)
image_vector()
# print(np_img[0, 0, 1])


plt.show()
