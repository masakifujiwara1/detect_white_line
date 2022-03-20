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
import math
#from keras.preprocessing import image
# matplotlib inline

path = "/home/fmasa/catkin_ws/src/detect_white_line/image/tsukuba/"
dir = "/home/fmasa/catkin_ws/src/detect_white_line/image/"
tsudanuma = "/home/fmasa/catkin_ws/src/detect_white_line/image/tsudanuma/"

#img = cv2.imread(dir + "20220314_152353.jpg")
# img = cv2.imread(path + "frame_233.jpg")
img = cv2.imread(tsudanuma + "1101_frame_1537.jpg")
# img = cv2.imread(dir + "tsudanuma001.jpeg")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
np_img = np.array(img2)
img_copy = img.copy()

# hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
# lower_white = np.array([0, 0, 100])
# upper_white = np.array([180, 45, 255])

# mask_white = cv2.inRange(hsv, lower_white, upper_white)
# res_white = cv2.bitwise_and(img_copy, img_copy, mask=mask_white)

# plt.figure(figsize=(5, 5))
# plt.xticks([]), plt.yticks([])
# plt.imshow(res_white)

# plt.figure(figsize=(5, 5))
# # img2 = img[:, :, ::-1]
# plt.xticks([]), plt.yticks([])
# plt.imshow(img[:, ::-1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_white = cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("image/gray.jpg", gray)
# img_p = gray

plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(gray)
plt.gray()

# h, w = gray.shape[0], gray.shape[1]
# print(h, w)

# edges = cv2.Canny(gray, 50, 150)
# cv2.imwrite("image/edges.jpg", edges)
# img_pp = cv2.imread("image/edges.jpg")

kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gray_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
gray_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=5)

lapsian = cv2.Laplacian(gray, cv2.CV_64F)

# edges = np.sqrt(gray_x**2 + gray_y**2)
edges = cv2.Canny(gray, 150, 300, L2gradient=True)
img_pp = edges
# abs_img_pp = np.absolute(img_pp)
# img_pp = np.uint8(abs_img_pp)

plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(img_pp)

# print(type(img), type(edges))
# template matching


def template(dir_path):
    template = cv2.imread(dir_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)


# template(dir + "template001.png")
# template(dir + "template002.png")
def calc_haarlike(crop_img):
    crop_img = crop_img[:, ::-1]

    threshold = 10
    rect_w = 8
    pattern_w = rect_w // 2
    width = crop_img.shape[0]

    peak_index = 0
    max_value = 0

    for index in range(width-rect_w):
        a1 = np.mean(crop_img[index: index+pattern_w, :])
        a2 = np.mean(crop_img[index+pattern_w: index+rect_w, :])
        H = a1-a2
        print(index, H)
        plt.plot(index, H, label="haar-like")
        if max_value < H and H - max_value > threshold:
            max_value = H
            peak_index = index

    index = width - peak_index + rect_w
    return index if max_value > 0 else -1


def candidate_extraction(img, gray):
    h, w = img.shape[0], img.shape[1]
    wh = 8
    window_pos = np.array([
        [[236, 40], [244, 440]],
        [[400, 40], [408, 440]]
    ])
    for i, ((x1, y1), (x2, y2)) in enumerate(window_pos):
        crop_img = gray[y1: y2, x1: x2]
        cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 1)
        peak_index = calc_haarlike(crop_img)
        cv2.circle(img2, (x2-4, peak_index-40), 4, (0, 0, 255), thickness=2)


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
            print(ass, ass_T)
            ass_res = ass * ass_T
            ass_res = (1/(num_pixel-1)) * ass_res
            # print(ass_res)
            res += ass_res
            # print(ass_res)
    print(res)

    Rp_val, Rp_vec = LA.eig(res)
    print(Rp_val)
    print(Rp_vec)


def hough():
    lines = cv2.HoughLinesP(img_pp,
                            rho=1,
                            theta=np.pi/360,
                            threshold=70,
                            minLineLength=50,
                            maxLineGap=20)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 赤線を引く
        red_line_img = cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)


hough()
# candidate_extraction(img, gray)
plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(img2)
# image_vector()
# print(np_img[0, 0, 1])


plt.show()
