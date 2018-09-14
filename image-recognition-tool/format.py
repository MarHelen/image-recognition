# format.py

import numpy as np
import imutils
import cv2
import argparse
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def put_text(img, text):
    """
    this function put text on the image
    in rectangular frame
    """

    fontFace = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1
    thickness = 2

    # colors for frame and label
    darc_orange = (0, 140, 255)
    light_orange = (0, 165, 255)
    opacity = 0.5

    img_change = img.copy()
    img_copy = img.copy()

    # get custom text size
    text_size, text_height = cv2.getTextSize(text, fontFace,
                                             fontScale, thickness)
    # text frame position
    x1 = int(img.shape[1] * 0.9) - text_size[0]
    y1 = int(img.shape[0] * 0.1)
    x_shift = int(text_size[0] * 1.1)
    y_shift = int(text_size[1] * 1.5)

    # frame drawing
    img_change = cv2.rectangle(img_change, (x1, y1),
                               (x1 + x_shift, y1 + y_shift),
                               darc_orange, 5, lineType=cv2.LINE_4)
    # filling drawing
    img_change = cv2.rectangle(img_change, (x1, y1),
                               (x1 + x_shift, y1 + y_shift),
                               light_orange, cv2.FILLED,
                               lineType=cv2.LINE_4)

    # make rectangular tranparent
    cv2.addWeighted(img_change, opacity, img_copy, 1 - opacity, 0, img_copy)

    # define text shift
    org = (x1 + int(text_size[0] * 0.05), y1 + int(text_size[1] * 1.1))

    cv2.putText(img_copy, text, org, fontFace, fontScale,
                (0, 0, 0), thickness)

    return img_copy


def save_results(image_name, image, image_horizon, image_sun,
                 horizon_flag, sun_flag, daytime):
    # fill result pdf file

    content = []
    output_file_name = image_name[:-4] + '.pdf'
    output_file = PdfPages(output_file_name)

    # adjust spacing
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=1, wspace=1, hspace=None)

    # build content for result pdf
    # [image, image_title, image_text]
    # main page
    text = 'Image \"' + image_name + '\" processing'
    content.append([image, None, text])

    # conditions page
    text = 'Conditions: ' + daytime
    content.append([image, text, None])

    # horizon page
    text = 'horizon detected' if horizon_flag else 'horizon line is unclear'
    content.append([image_horizon, text, None])

    # sun page
    if daytime != 'nighttime':
        text = 'sun located' if sun_flag else 'no sun spot located'
        content.append([image_sun, text, None])

    # save result into pdf format
    for item in content:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # if title present
        if item[2]:
            fig.suptitle(item[2], fontsize=12, fontweight='bold')

        # if text present
        if item[1]:
            ax.text(image.shape[1] * 2.5, image.shape[0], item[1],
                    style='italic',
                    bbox={'facecolor': 'orange', 'alpha': 0.5, 'pad': 5})

        plt.imshow(imutils.opencv2matplotlib(item[0]))
        fig.tight_layout()
        plt.axis('off')
        output_file.savefig()

    output_file.close()

    # save result into image format
    png_filename = image_name[:-4] + '_processed.png'

    res_img = image
    for item in content:
        # if text present
        if item[1]:
            image_add = put_text(item[0], item[1])
            res_img = np.concatenate((res_img, image_add), axis=0)

    # plt.imshow(imutils.opencv2matplotlib(res_img))
    cv2.imwrite(png_filename, res_img)

    print 'check', output_file_name, 'or', png_filename, 'for results'
