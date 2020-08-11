from math import sqrt, pow
from dataclasses import dataclass, astuple
import numpy as np

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance(self, other) -> float:
        return sqrt(pow(other.x - self.x, 2) + pow(other.y - self.y, 2))
    
    def as_int(self) -> 'Point':
        return Point(int(self.x), int(self.y))

@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def center(self) -> Point:
        return Point(self.xmin + ((self.xmax-self.xmin)/2), 
                     self.ymin + ((self.ymax-self.ymin)/2))

    def contains(self, point: Point) -> bool:
        if int(self.xmin) > int(point.x) or int(self.xmax) < int(point.x):
            return False
        if int(self.ymin) > int(point.y) or int(self.ymax) < int(point.y):
            return False
        return True

    def overlaps(self, other) -> bool:
        if int(self.xmin) > int(other.xmax) or int(self.xmax) < int(other.xmin):
            return False
        if int(self.ymin) > int(other.ymax) or int(self.ymax) < int(other.ymin):
            return False
        return True

    def width(self) -> float:
        return self.xmax-self.xmin

    def height(self) -> float:
        return self.ymax-self.ymin

    def area(self) -> float:
        return self.width() * self.height()

    def __getitem__(self, key):
        if key == 0:
            return self.xmin
        if key == 1:
            return self.ymin
        if key == 2:
            return self.xmax
        if key == 3:
            return self.ymax
        raise IndexError

    @staticmethod
    def as_nparray(rect):
        return np.array(Rect.as_array(rect), dtype=np.float_)

    @staticmethod
    def as_array(rect):
        return [rect.xmin, rect.ymin, rect.xmax, rect.ymax]

    @staticmethod
    def from_cvrect(x, y, w, h):
        rx = w/2
        ry = h/2
        return Rect(x-rx, y-ry, x+rx, y+ry)

    @staticmethod
    def nms_merge(rects, overlapThresh=0.3):
        # this function is from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        if len(rects) == 0:
            return []

        boxes = np.array(list(map(astuple, rects)))

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
            
        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:

            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return list(map(lambda a: Rect(*a), boxes[pick].astype("int")))
