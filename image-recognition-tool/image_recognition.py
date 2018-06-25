# !/usr/bin/env python
# main.py

import os
import numpy as np
import imutils
import cv2
import argparse
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sun import sun_detection
from horizon import horizon_detection
from horizon import draw_lines
from format import save_results

scale = 4
# maxRadius = 90


def parse_image_path():
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='image recognition tool')
    parser.add_argument("-i", "--image", required=True,
                        help="path to the *.png image file")
    # ap.add_argument("-d", "--dark", required=False,
    #    help="dark image flag")
    args = vars(parser.parse_args())

    if not os.path.isfile('./' + args["image"]):
        parser.error('specified file or folder does not exist')

    if not args["image"].endswith(".png"):
        parser.error('specified file has a wrong extension, \
                      only *.png images are allowed')

    return args["image"]


if __name__ == "__main__":

    img_file = parse_image_path()

    # load the image, convert it to grayscale, and blur
    image = cv2.imread('./' + img_file)
    resized = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # (5,5)

    # create image copies for further processing
    full_for_hybrid = image.copy()
    full_for_sun = image.copy()

    # build horizon
    (horizon, result_type), daytime = horizon_detection(blurred)
    if result_type == 1:
        full_for_hybrid = draw_lines(full_for_hybrid, horizon, scale)
        print 'horizon detected'
    elif horizon and len(horizon) > 3:
        full_for_hybrid = draw_lines(full_for_hybrid, horizon, scale)
        print 'possible horizon detected'
    else:
        print 'no horizon detected'

    print 'it is', daytime

    # detect sun
    maxRadius = 90
    sun_center = None
    if daytime != 'nighttime':
        sun_center = sun_detection(gray)
        if sun_center is not None:
            sun_center = (scale * sun_center[0], scale * sun_center[1])
            cv2.circle(full_for_sun, sun_center, maxRadius, (0, 0, 255), 4)
            print 'sun spot located ', sun_center
        else:
            print 'no sun spot located'
    else:
        print 'no sun spot located because of nighttime'

    save_results(img_file, image, full_for_hybrid, full_for_sun,
                 result_type, sun_center, daytime)
