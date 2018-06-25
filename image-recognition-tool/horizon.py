# horizon.py

from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import random
import os
from matplotlib import pyplot as plt


def calc_angle(first, second):
    """
    calculates angle between x-axis and (first, second) line
    """
    if second[0] == first[0]:
        return np.pi/2 if second[1] > first[1] else -np.pi/2

    tg = float(second[1]-first[1])/(second[0]-first[0])

    return np.arctan(tg)


def check_pos(a, a_max, mult):
    return min(a_max-1, a * mult)


def draw_lines(img, lines, scale):
    """
    this function draw lines on a picture, according to scale
    first point from the left is added in case of absence
    last point from the right is added in case of absence
    returns updated image
    """

    if lines is not None and len(lines) > 0:
        # calculate angle changes between first 2 points
        # and find start y in accordance to this angle
        angle = calc_angle(lines[0], lines[-1])
        check_pos(lines[0][1], img.shape[0], scale)
        start_y = int(check_pos(lines[0][1], img.shape[0], scale) -
                      check_pos(lines[0][0], img.shape[1], scale) *
                      np.tan(angle))
        start = (0, start_y)

        for line in lines:
            cv2.line(img, start, (check_pos(line[0], img.shape[1], scale),
                                  check_pos(line[1], img.shape[0], scale)),
                     (0, 255, 0), 5)

            if len(line) > 2:
                cv2.line(img, (check_pos(line[0], img.shape[1], scale),
                               check_pos(line[1], img.shape[0], scale)),
                              (check_pos(line[2], img.shape[1], scale),
                               check_pos(line[3], img.shape[0], scale)),
                         (0, 255, 0), 5)

                start = (check_pos(line[2], img.shape[1], scale),
                         check_pos(line[3], img.shape[0], scale))
            else:
                start = (check_pos(line[0], img.shape[1], scale),
                         check_pos(line[1], img.shape[0], scale))

        y_first = check_pos(lines[0][1], img.shape[0], scale)
        y_last = (start[1] - y_first) * img.shape[1] / start[0] + y_first
        x_last = img.shape[1] - 1

        cv2.line(img, start, (x_last, y_last), (0, 255, 0), 6)

    return img


def contrast_changing(gray_img, contrast_delta):
    # decrease contrast
    n, m = gray_img.shape
    contrast_high = 255

    for i in range(n):
        for j in range(m):
            gray_img[i][j] = gray_img[i][j] if (gray_img[i][j] < 220) else 220

    return gray_img


def find_countour(img):
    cnt = []
    for j in range(10, img.shape[1] - 10):
        for i in range(10, img.shape[0] - 10):
            if (img[i+1][j] == 0) and img[i][j] == 255:
                cnt.append([j, i])

    return cnt


def filter_contour_compare_ctg(first, second, third, arctg_delta=np.pi/60):
    """
    compare changes between angles of triangles on bases of
    parallel x axis line and:
    - first point, second point ctg_1
    - first point, third point ctg_2
    """
    arctg_1 = calc_angle(first, second) + np.pi/2
    arctg_2 = calc_angle(first, third) + np.pi/2
    curr_arctg = calc_angle(second, third)

    if abs(arctg_1-arctg_2) < arctg_delta:
        return 1
    else:
        return 0


def filter_contour(cnt, delta):
    """
    filter the longest increasing point subsequence in contour
    """
    i = 0
    # track the list of unfit points to start a new subsequence from
    unfit = [1 for i in range(len(cnt))]
    max_len_subs = []
    while i < len(cnt) and len(max_len_subs) < (len(cnt) - i):
        j = i
        curr = []

        while j < len(cnt) and len(max_len_subs) < (len(cnt) - j + len(curr)):
            if len(curr):
                last_added = cnt[curr[-1]]
                first_added = cnt[curr[0]]
                if len(curr) > 2:
                    prev_added = cnt[curr[-2]]

                    # check 2nd from the end point and replace the last
                    # if observed point fits better
                    if (cnt[j][1] - last_added[1]) * \
                        (last_added[1] - prev_added[1]) < 0 \
                        and (cnt[j][1] - last_added[1]) > delta / 2 and \
                        (last_added[1] - prev_added[1]) > delta / 2 and \
                        filter_contour_compare_ctg(first_added,
                                                   prev_added, cnt[j]):
                        curr[-1] = j
                        unfit[j] = 0

                    # compare angle changes from 1st added point and x-axis
                    # add point if it's not changed more than on delta
                    elif filter_contour_compare_ctg(first_added,
                                                    last_added, cnt[j]):
                        curr.append(j)

                elif (abs(last_added[-1] - cnt[j][1]) < delta):
                    curr.append(j)
                    unfit[j] = 0

            else:
                curr.append(j)
                unfit[j] = 0

            if j != curr[-1] and len(max_len_subs) < len(cnt)-j:
                unfit[j] = 1

            j += 1

        max_len_subs = curr if len(curr) > len(max_len_subs) else max_len_subs

        if 1 in unfit:
            i = unfit.index(1)
            unfit[i] = 0
        else:
            i = len(cnt)

    return max_len_subs


def hybrid_find_y_average(horizont, x, point, i, n, delta):
    """
    calculate average y by 20 subsequent points in thresh horizont list
    calculate angle between first and last added points line and x=0
    return also current index and last observed point
    """
    avg = 0
    m = i + 20
    summ = point[1]
    count = 1
    first_point = point
    while x > point[0] and i < min(n-1, m):
        current_point = horizont[i]
        summ += current_point[1]
        count += 1
        point = horizont[i]
        i += 1

    x_delta = 1 if point[0] == first_point[0] else point[0] - first_point[0]
    avg_angle = np.arctan(float(point[1] - first_point[1]) / x_delta)
    avg = summ / count

    return i, avg, avg_angle, point


def line_bright_check(img, x, y):
    """
    check if the line divide image on contrast zones with more intensive above
    and less intensive below
    """
    if (y-2 > 0 and y+2 < img.shape[0]) and img[y-2][x] + 30 > img[y+2][x] \
       and img[y-1][x] + 30 > img[y+1][x]:
        return 1
    else:
        return 0


def hybrid_add_line_check(lines, x, y, x2, y2, avg,
                          avg_angle, gray_image, delta):
    """
    check if next line in delta coridor around thresh horizont line
    check if next line start after previous
    add observing line only if it's start point is (y_delta/2)
    close to prev added end point by Y
    """
    len_delta = gray_image.shape[1] * 0.03

    if lines:
        angle = calc_angle(lines[-1], (x, y))
        if abs(angle - avg_angle) > np.pi/30:
            return 0

    if (avg - y > delta) and not lines:
        return 0

    if lines and abs(y - lines[-1][3]) > delta:
        return 0

    if lines and abs(y - lines[-1][3]) > delta / 2 and x2 - x < len_delta:
        return 0

    if lines and x < lines[-1][2] + 3:
        return 0

    return 1


def hybrid_replace_line_check(lines, x, y, x2, y2, avg, avg_angle, image):
    """
    if new observing point is closer to avg
    and it's start X less than added point end X
    exchange last added to a new observing
    """
    delta = 0.05 * image.shape[0]
    if not lines:
        return 0

    last_x1, last_x2 = lines[-1][0], lines[-1][2]
    last_y1, last_y2 = lines[-1][1], lines[-1][3]

    if y <= last_y1 and x2-x > 2 * (last_x2-last_x1):
        return 1

    if len(lines) > 1:
        prev_x1, prev_x2 = lines[-2][0], lines[-2][2]
        prev_y1, prev_y2 = lines[-2][1], lines[-2][3]
        if 2 * abs(y - prev_y1) < abs(last_y1 - prev_y1):
            return 1

    if abs(y - avg) > abs(lines[-1][1] - avg):
        if x2 - x < 2 * abs(lines[-1][0] - lines[-1][2]):
            return 0
        elif abs(y - avg) > abs(lines[-1][1] - avg) + delta:
            return 0

    # if contrast change on a new suggested line is twice less
    if lines and y-1 > 0 and y + 1 < image.shape[0]:
        lines_x = lines[-1][0]
        lines_y = lines[-1][1]

        new_line_length = int(image[lines_y - 2][lines_x]) - \
            int(image[lines_y][lines_x])
        old_line_length = int(image[y - 2][x]) - int(image[y][x])
        if lines_y - 2 > 0 and lines_y + 1 < image.shape[0]:
            if old_line_length * 2 > new_line_length:
                return 0

    return 1


def hybride_build(horizon, im_shape, Hough_lines, img):
    """
    compare HoughLines results and threshhold results
    keep those HoughLines wich start points are closest to
    threshold points by Y
    """
    horizon_lines = []
    n = len(horizon)
    edge_point = horizon[0]
    i = 1
    avg = edge_point[1]
    avg_angle = calc_angle(horizon[0], horizon[-1])
    y_delta = im_shape[1]*0.03

    if Hough_lines is not None:
        for line in Hough_lines:
            for x, y, x2, y2 in line:
                if x > edge_point[0]:
                    i, avg, avg_angle, edge_point = \
                        hybrid_find_y_average(horizon, x, edge_point,
                                              i, n, y_delta)

                    avg = avg if avg > 0 else edge_point[1]

                if y < (avg + y_delta) and y2 < (avg + y_delta) and \
                    line_bright_check(img, x, y) and \
                        line_bright_check(img, x2, y2):

                    if hybrid_add_line_check(horizon_lines, x, y, x2, y2,
                                             avg, avg_angle, img, y_delta):
                        horizon_lines.append([x, y, x2, y2])

                    elif hybrid_replace_line_check(horizon_lines, x, y, x2,
                                                   y2, avg, avg_angle, img):
                        horizon_lines[-1] = [x, y, x2, y2]

    return horizon_lines


def build_hough_lines(canny):
    """
    this function run Hough probobalistic transform on canny image map
    reverse line with wrong line directions (x2 < x1)
    hand sort all lines in ascending order
    """
    rho = 2
    theta = np.pi/180
    threshold = 40
    minLineLength = 20
    maxLineGap = 5
    Hough_lines = cv2.HoughLinesP(canny, rho, theta, threshold)

    if Hough_lines is not None:

        n = Hough_lines.shape[0]
        for i in range(n):
            if Hough_lines[i][0][0] > Hough_lines[i][0][2]:
                Hough_lines[i][0][0], Hough_lines[i][0][2] = \
                    Hough_lines[i][0][2], Hough_lines[i][0][0]

                Hough_lines[i][0][1], Hough_lines[i][0][3] = \
                    Hough_lines[i][0][3], Hough_lines[i][0][1]
            sort_ind = Hough_lines[:, 0, 0].argsort()
        Hough_lines = Hough_lines[sort_ind]

    return Hough_lines


def build_threshold(gray):
    """
    this function buids Otsu binary threshold for image
    and also run morphological filters for the result to smooth
    noisy zone
    """
    thresh_val = 0
    maxValue = 255
    (thresh_param, thresh) = cv2.threshold(gray, thresh_val, maxValue,
                                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print 'thresh param', thresh_param

    # morhological filters
    thresh = cv2.erode(thresh, None, iterations=4)
    thresh = cv2.dilate(thresh, None, iterations=3)

    return thresh_param, thresh


def check_daytime(param):
    """
    this function returns possible daytime haracteristic
    in accordance to calculated image threshold parameter
    """

    if param > 120:
        return 'daytime'
    elif param > 78:
        return 'dusk/shade'
    else:
        return 'nighttime'


def horizon_detection(img):
    """
    this function build resulting set of lines for horizon drawing
    it calls for canny to build image map
    then generate hough lines probobalistic transform
    then build threshold image, enject contour as possible horizon
    then build hybrid set of line among hough lines result
    the closest to hough lines horizon-possible set of lines
    returns hybrid line result if that exists, otherwise contour points
    returingn type: (result, flag)
    flag = 1 if hybrid exists
    flag = 0 if contour points returns
    """
    (mu, sigma) = cv2.meanStdDev(img)
    canny = cv2.Canny(img, 30, mu + sigma/4)

    # contrast_delta = 80

    # build hough probobalistic transform
    hough_lines = build_hough_lines(canny)

    # gray = contrast_changing(blurred,contrast_delta)

    # adjust sharpness
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)

    # build threshold image
    thresh_param, thresh = build_threshold(img)

    daytime = check_daytime(thresh_param)

    point_delta = 0.1 * img.shape[0]
    cont = find_countour(thresh)

    indx_list = filter_contour(cont, point_delta)

    horizon_thresh = []
    hybride = []
    if indx_list is not None and len(indx_list):
        horizon_thresh = [[cont[indx_list[0]][0], cont[indx_list[0]][1]]]
        added = [cont[i] for i in indx_list]
        horizon_thresh += added

        hybride = hybride_build(horizon_thresh, img.shape, hough_lines, img)

    result = (hybride, 1) if hybride else (horizon_thresh, 0)

    return result, daytime
