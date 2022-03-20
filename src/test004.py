# from distutils.log import info
# from tkinter import image_names
# from types import GeneratorType

from multiprocessing.context import assert_spawning
from statistics import mode
from turtle import shape, up
from attr import assoc
from cv2 import cvtColor, sort, threshold
import std_msgs
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import csv
from scipy import stats
# from keras.preprocessing import image
# matplotlib inline

path = "/home/fmasa/catkin_ws/src/detect_white_line/image/tsukuba/"
dir = "/home/fmasa/catkin_ws/src/detect_white_line/image/"
tsudanuma = "/home/fmasa/catkin_ws/src/detect_white_line/image/tsudanuma/"
tsudanuma_usb = "/home/fmasa/catkin_ws/src/detect_white_line/image/tsudanuma_usbcam_1030/"
day = "2021-10-30-10-"

size = (640, 480)
GAMMA = 2.5
over = 0
'''
select dir and image
'''
# img = cv2.imread(path + "frame_246.jpg")

# 明るいimage
# img = cv2.imread(tsudanuma_usb + day + "41-44/frame000016.jpg")

# 暗いimage
# img = cv2.imread(tsudanuma_usb + day + "54-09/frame000035.jpg")

# img = cv2.imread(tsudanuma_usb + day + "48-03/frame000026.jpg")
img = cv2.imread(tsudanuma_usb + day + "54-09/frame000062.jpg")
# img = cv2.imread(tsudanuma + "1101_frame_1296.jpg")
# img = cv2.imread(dir + "20220314_152251.jpg")
# img = cv2.imread(dir + "tsudanuma001.jpeg")
# img = cv2.resize(img, size)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_gamma_img(gamma, img):
    gamma_cvt = np.zeros((256, 1), dtype=np.uint8)
    for i in range(256):
        gamma_cvt[i][0] = 255*(float(i)/255)**(1.0/gamma)
    return cv2.LUT(img, gamma_cvt)


# img2 = create_gamma_img(GAMMA, img2)
# plt.imshow(img2)

hsv_calc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# calc_hist
hist = cv2.calcHist([hsv_calc], [0], None, [256], [0, 256])
# plt.plot(hist)
# mode
mode1, count = stats.mode(hsv_calc.ravel())
print("mode", mode1)
# median
median = np.median(hsv_calc)
print("median", median)
# average
mean = hsv_calc.mean()
print("avg", mean)


np_img = np.array(img2)
img_copy = img.copy()

img_copy = cv2.medianBlur(img_copy, 5)
hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
# tsukuba 60, 0, 150 : tsudanuma 60, 0, 240 : tsudanuma shadowb 60, 0, 230
lower_white = np.array([60, 0, 230])
# tsukuba 180, 45, 255 : tsudanuma 200, 45 ,255 : tsudanuma shadow 210, 60, 255
upper_white = np.array([200, 45, 255])

lower_silver = np.array([0, 0, 75])
upper_silver = np.array([0, 0, 200])

mask_white = cv2.inRange(hsv, lower_white, upper_white)
res_white = cv2.bitwise_and(img_copy, img_copy, mask=mask_white)

mask_silver = cv2.inRange(hsv, lower_silver, upper_silver)
res_silver = cv2.bitwise_and(img_copy, img_copy, mask=mask_silver)

plt.figure(figsize=(5, 5))
plt.xticks([]), plt.yticks([])
plt.imshow(res_white)
plt.show()


# plt.figure(figsize=(5, 5))
# # img2 = img[:, :, ::-1]
# plt.xticks([]), plt.yticks([])
# plt.imshow(img[:, ::-1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_white = cv2.cvtColor(res_white, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("image/gray.jpg", gray)
img_p = gray_white
gray = gray_white

# plt.figure(figsize=(5, 5))
# plt.xticks([]), plt.yticks([])
# plt.imshow(gray_white)
# plt.gray()
# plt.show()

# h, w = gray.shape[0], gray.shape[1]
# print(h, w)

# edges = cv2.Canny(gray, 50, 150)
# cv2.imwrite("image/edges.jpg", edges)
# img_pp = cv2.imread("image/edges.jpg")

# abs_img_pp = np.absolute(img_pp)
# img_pp = np.uint8(abs_img_pp)

# plt.figure(figsize=(5, 5))
# plt.xticks([]), plt.yticks([])
# plt.imshow(img_pp)

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
        # print(index, H)
        # plt.plot(index, H, label="haar-like")
        # line = [str(index), str(H)]
        # with open(dir + 'csv/haarlike.csv', 'a') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(line)

        if max_value < H and H - max_value > threshold:
            max_value = H
            peak_index = index

    index = width - peak_index + rect_w
    return index if max_value > 0 else -1


def candidate_extraction(img):
    x = []
    y = []
    before_y = []
    regist = []
    global over
    h, w = img.shape[0], img.shape[1]
    wh = 8
    window_pos = np.array([
        [[100, 200], [108, 470]],
        [[168, 200], [176, 470]],
        [[236, 200], [244, 470]],
        [[316, 200], [324, 470]],
        [[400, 200], [408, 470]],
        [[470, 200], [478, 470]],
        [[540, 200], [548, 470]]
    ])
    for i, ((x1, y1), (x2, y2)) in enumerate(window_pos):
        crop_img = gray[y1: y2, x1: x2]
        cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 1)
        peak_index = calc_haarlike(crop_img)
        cv2.circle(img2, (x2-4,  470 - peak_index),
                   4, (0, 0, 255), thickness=2)
        x.append(x2-4)
        y.append(-1*(470 - peak_index))
        before_y.append(470 - peak_index)

    # 外れ値除外 IQR 2, 4, 6
    sort_y = sorted(before_y)
    # print(sort_y)
    range_IQR = int(sort_y[5]) - int(sort_y[1])
    # print(range)
    lim_upper = -1 * (int(sort_y[5]) + (range_IQR * 1.5))
    lim_lower = -1 * (int(sort_y[1]) - (range_IQR * 1.5))
    print(lim_upper, lim_lower)
    for i in range(len(x)):
        y_ass = int(y[i])
        if y_ass <= lim_upper or y_ass >= lim_lower:
            print("外れ値検出：x", i)
            over += 1
            regist.append(y_ass)
    # del element of x and y list
    for j in range(len(regist)):
        local = regist.index(regist[j])
        del x[local]
        del y[local]

    # print(x, y)
    return x, y


def calc_line(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    a = ((np.dot(x, y) - y.sum() * x.sum()/n) /
         ((x ** 2).sum() - x.sum()**2 / n))
    b = (y.sum() - a * x.sum())/n
    # print(a, b)
    return a, b


def calc_angle(y1, y2):
    # print(y1, y2)
    if y1 > y2:
        a = y1 - y2
    else:
        a = y2 - y1
    tan = a / 640
    deg = math.degrees(math.atan(tan))
    print("degree:", round(deg, 2), "度")


def draw_line(a, b):
    y2 = -1 * (a * 640 + b)
    b = int(b)
    y2 = int(y2)
    y1 = -1 * b
    calc_angle(y1, y2)
    cv2.line(img2, (0, y1), (640, y2), (0, 255, 0), 2)


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
                            threshold=100,
                            minLineLength=200,
                            maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 赤線を引く
        red_line_img = cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 1)


# hough()
x, y = candidate_extraction(gray)
if over < 3:
    a, b = calc_line(x, y)
# print(a, b)
    draw_line(a, b)
# plt.figure(figsize=(5, 5))
# plt.xticks([]), plt.yticks([])
plt.imshow(img2)
# image_vector()
# print(np_img[0, 0, 1])


plt.show()
