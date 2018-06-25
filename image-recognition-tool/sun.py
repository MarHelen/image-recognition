# sun_detection.py

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def check_inside_circle(x, y, x_c, y_c, r):
    """
    check if point inside of circle
    """
    return (x - x_c) * (x - x_c) + (y - y_c) * (y - y_c) < r * r


def check_circle(circle, thresh):
    """
    check if at least 70% of thresholded by 250 value
    sement got into consider circle
    """
    percent = 0.7

    # get those indicies of thresh image where value is 255
    indx = np.where(thresh == 255)

    # amount of all pixels with 255 value
    n = indx[0].shape[0]
    amount = 0

    # calculate amount of 255-val pixels inside consider circle
    for i in range(n):
        x = indx[1][i]
        y = indx[0][i]

        if check_inside_circle(x, y, circle[0], circle[1], circle[2]):
            amount += 1

    return 1 if float(amount)/n > percent else 0


def sun_detection(img):
    scale = 4
    result_circle = None

    img = cv2.medianBlur(img, 5)

    # binary threshold building for 250 and up zone
    thresh = cv2.threshold(img, 250, 255,  cv2.THRESH_BINARY)[1]

    # check if there any brigh zone
    # return None if no
    if not np.any(thresh == 255):
        return result_circle

    # build a wider threshold
    thresh_2 = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO)[1]

    # run morphological filters
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # HoughCircles parameters
    param1 = 70  # 200 #50 #7
    param2 = 50  # 100 #100 #20
    minRadius = 5  # 1 #40
    maxRadius = 30  # 5 #80
    circles = cv2.HoughCircles(thresh_2, cv2.HOUGH_GRADIENT, 1, 20,
                               param1, param2, minRadius, maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if thresh[i[1]][i[0]] == 255:
                if check_circle(i, thresh):
                    result_circle = (i[0], i[1])
                    break

    return result_circle
