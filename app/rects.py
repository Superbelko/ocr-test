from typing import Optional, List
import numpy as np
import cv2 as cv

from app.common import Rect
from app.debug import COLORS



def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)

    # return the edged image
    return edged

# fancy but nope, we don't have uniform color
def norm_color_test(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    h,s,v = cv.split(hsv)

    # 145<whites<165 white'ish range in h of HSV colorspace
    h[h<145]=0
    h[h>165]=0

    # normalize, do the open-morp-op
    normed = cv.normalize(h, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3,3))
    opened = cv.morphologyEx(normed, cv.MORPH_OPEN, kernel)

    return opened


def find_rects_white(image: np.ndarray) -> List[np.ndarray]:
    """Detects rectangular contours in an image
    @returns list containing arrays of rect corners
    """

    raise NotImplementedError()

    #gray = norm_color_test(image.copy())
    gray = cv.cvtColor(image.copy(), cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    #edged = cv.Canny(gray, 70, 180) # this numbers is hand picked guess from a few photos
    edged = auto_canny(gray) # this numbers is hand picked guess from a few photos

    # split to HSV, then pick up rouhly any white color zeroing out the rest
    hsv = cv.cvtColor(image.copy(), cv.COLOR_RGB2HSV)
    h,s,v = cv.split(hsv)
    h[h<145] = 0
    h[h>165] = 0
    #h = cv.GaussianBlur(h, (5, 5), 0)
    normed = cv.normalize(h, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(5,5))
    opened = cv.morphologyEx(normed, cv.MORPH_OPEN, kernel)

    # now find white regions contours
    whites = cv.findContours(opened, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    whites.sort(key=cv.contourArea, reverse=True)
    whites = [cnt for cnt in whites if cv.contourArea(cnt) > 15] # 15px contour area, basically cnt>=4x4

    whiterects = []
    for i in whites:
        rect = cv.minAreaRect(i)
        w,h = rect[1]
        if w*h > 150: # 150px area, or rougly 12x12 pixels
            whiterects.append(rect)

    #cv.drawContours(image, whites, -1, COLORS[2 % len(COLORS)], 2)
    #cv.imshow('test', image)
    #cv.waitKey()

    whites = list(map(lambda r: Rect.from_cvrect(*r[0], *r[1]), whiterects))



    #cv.imshow('test', edged)
    #cv.waitKey()

    contours = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours.sort(key=cv.contourArea, reverse=True)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 15] # 15px contour area, basically cnt>=4x4

    rects = list(map(cv.minAreaRect, contours))
    boxes = list(map(lambda r: Rect.from_cvrect(*r[0], *r[1]), rects))

    # filter non overlapping contours
    for i in reversed(range(len(boxes))):
        overlaps = False
        for wbox in whites:
            if wbox.overlaps(boxes[i]):
                overlaps = True
                break
        if not overlaps:
            boxes.pop(i)

    boxes = Rect.nms_merge(boxes)

    for i in range(len(contours)):
        #peri = cv.arcLength(contours[i], True)
        #approx = cv.approxPolyDP(contours[i], 0.02 * peri, True)
        rect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        #cv.drawContours(image, [box], -1, COLORS[i % len(COLORS)], 2)
        #cv.putText(image, f'{i}: {cv.contourArea(contours[i])}px', (int(rect[0][0]), int(rect[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i % len(COLORS)], 1)

    #cv.drawContours(image, contours, -1, COLORS[1], 2)

    for b in boxes:
        cv.line(image, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymin)), (0, 255, 255), 2)
        cv.line(image, (int(b.xmin), int(b.ymax)), (int(b.xmax), int(b.ymax)), (0, 255, 255), 2)
        cv.line(image, (int(b.xmin), int(b.ymin)), (int(b.xmin), int(b.ymax)), (0, 255, 255), 2)
        cv.line(image, (int(b.xmax), int(b.ymin)), (int(b.xmax), int(b.ymax)), (0, 255, 255), 2)

    stacked = np.hstack( (cv.cvtColor(edged, cv.COLOR_GRAY2RGB), cv.cvtColor(opened, cv.COLOR_GRAY2RGB), image))
    cv.namedWindow('test', 0)
    cv.imshow('test', stacked)
    cv.waitKey()

    cv.imwrite('dump.jpg', stacked)


    return boxes or list()


def find_rects(image: np.ndarray) -> List[np.ndarray]:
    """Detects rectangular contours in an image
    @returns list containing arrays of rect corners
    """

    gray = cv.cvtColor(image.copy(), cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    #edged = cv.Canny(gray, 70, 180) # this numbers is hand picked guess from a few photos
    edged = auto_canny(gray)

    contours = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours.sort(key=cv.contourArea, reverse=True)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 15] # 15px contour area, basically cnt>=4x4

    rects = list(map(cv.minAreaRect, contours))
    boxes = list(map(lambda r: Rect.from_cvrect(*r[0], *r[1]), rects))

    boxes = Rect.nms_merge(boxes)

    return boxes or list()
