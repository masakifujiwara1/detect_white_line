#!/usr/bin/env python
# -*- coding: utf-8 -*-

from geometry_msgs.msg import Twist
import numpy as np
import cv2
from sensor_msgs.msg import Image
import math
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError


class detect_white_line_node():
    def __init__(self):
        rospy.init_node('detect_white_line_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.callback)
        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.size = (640, 480)
        self.GAMMA = 2.5
        self.over = 0
        self.count_del = 0
        self.before_y = []
        self.x = []
        self.y = []
        self.deg = 0
        self.save = True
        self.center = 0
        self.Flag = False
        self.vel_pub = rospy.Publisher("white_vel", Twist, queue_size=10)
        self.vel = Twist()
        self.status = False
        self.count_end = 0

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def calc_haarlike(self, crop_img):
        crop_img = crop_img[:, ::-1]

        threshold = 10
        rect_h = 4
        pattern_h = rect_h // 2
        height = crop_img.shape[0]

        peak_index = 0
        max_value = 0

        for index in range(height-rect_h):
            a1 = np.mean(crop_img[index: index+pattern_h, :])
            a2 = np.mean(crop_img[index+pattern_h: index+rect_h, :])
            H = a1-a2

            if max_value < H and H - max_value > threshold:
                max_value = H
                peak_index = index

        index = height - peak_index + rect_h
        return index if max_value > 0 else -1

    def IQR(self):  # excecpt out of range value
        regist = []
        sort_y = sorted(self.before_y)
        range_IQR = int(sort_y[5]) - int(sort_y[1])
        lim_upper = -1 * (int(sort_y[5]) + (range_IQR * 1.5))
        lim_lower = -1 * (int(sort_y[1]) - (range_IQR * 1.5))
        for i in range(len(self.x)):
            y_ass = int(self.y[i])
            if y_ass <= lim_upper or y_ass >= lim_lower:
                print("Detect out of range value -> x", i)
                self.over += 1
                regist.append(y_ass)
        # del element of x and y list
        for j in range(len(regist)):
            local = self.y.index(regist[j])
            del self.x[local]
            del self.y[local]

    def candidate_extraction(self, gray, img2):
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
            peak_index = self.calc_haarlike(crop_img)

            calced_y = -1 * (470 - peak_index)
            if -471 <= calced_y <= -468:
                self.count_del += 1
            else:
                cv2.circle(img2, (x2-4,  470 - peak_index),
                           4, (0, 0, 255), thickness=2)
                self.x.append(x2-4)
                self.y.append(-1*(470 - peak_index))
                self.before_y.append(470 - peak_index)
        if self.count_del < 2:
            self.IQR()

    def calc_line(self):
        x = np.array(self.x)
        y = np.array(self.y)
        coe = np.polyfit(x, y, 1)
        return coe[0], coe[1]

    def calc_angle(self, y1, y2):
        if y1 > y2:
            z = y1 - y2
        else:
            z = y2 - y1
        tan = float(z) / 640
        self.deg = math.degrees(math.atan(tan))
        print("degree ->", round(self.deg, 2))

    def draw_line(self, a, b, img2):
        y2 = -1 * (a * 640 + b)
        self.center = -1 * (a * 320 + b)
        b = int(b)
        y2 = int(y2)
        y1 = -1 * b
        self.calc_angle(y1, y2)
        cv2.line(img2, (0, y1), (640, y2), (0, 255, 0), 5)
        cv2.circle(img2, (320,  int(self.center)),
                   6, (0, 0, 0), thickness=4)

    def callback_srv(self, data):
        resp = TriggerResponse()
        self.Flag = True
        self.status = True
        resp.message = "detect start"
        resp.success = True
        self.count_end = 0
        print(resp.message)
        return resp

    def control_move(self):
        if self.status:
            if self.count_end >= 1:
                self.vel.linear.x = 0.1
            else:
                self.vel.linear.x = 0.2
        else:
            self.vel.linear.x = 0.0
        self.vel_pub.publish(self.vel)

    def loop(self):
        self.over = 0
        self.count_del = 0
        self.before_y = []
        self.x = []
        self.y = []
        self.deg = 0
        self.center = 0

        if self.Flag:
            img = self.cv_image
            img2 = img.copy()
            img_copy = img.copy()

            img_copy = cv2.medianBlur(img_copy, 5)
            hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
            # tsukuba 60, 0, 150 : tsudanuma 60, 0, 240 : tsudanuma shadow 60, 0, 230
            lower_white = np.array([60, 0, 230])
            # tsukuba 180, 45, 255 : tsudanuma 200, 45 ,255 : tsudanuma shadow 210, 60, 255
            upper_white = np.array([200, 45, 255])

            res_white = cv2.inRange(hsv, lower_white, upper_white)

            cv2.imshow("mask", res_white)

            gray = res_white

            self.candidate_extraction(gray, img2)

            if self.over <= 3 and (not self.count_del > 5):
                a, b = self.calc_line()
                self.draw_line(a, b, img2)
                cv2.putText(img2, 'Detect white line!', (0, 100),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3, 8)
            else:
                cv2.putText(img2, 'Not detect white line', (0, 100),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 8)
            # status
            if self.center >= 300:
                cv2.putText(img2, 'status:STOP', (0, 50),
                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 255), 5, cv2.LINE_AA)

                self.count_end += 1
                self.control_move()
                if self.count_end >= 5:
                    self.status = False
                    self.control_move()
                    self.Flag = False
            else:
                cv2.putText(img2, 'status:GO', (0, 50),
                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5, cv2.LINE_AA)

                self.control_move()
            cv2.imshow("detect process", img2)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.loginfo('detect_white_line_node started')
    rg = detect_white_line_node()
    srv = rospy.Service("start_detect", Trigger, rg.callback_srv)
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
