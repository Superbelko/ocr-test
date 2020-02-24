from math import sqrt, pow
from dataclasses import dataclass
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
