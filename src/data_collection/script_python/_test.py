import numpy as np
from .tranformation import PointTransformation
from geometry import Point


class DiskTransformation(PointTransformation):
    """
    This is disk transformation for point
    """

    def __call__(self, p: Point) -> Point:
        theta = np.arctan2(p.y, p.x)
        r = np.sqrt(p.x**2 + p.y**2) + 1e-8
        new_x = (theta / np.pi) * np.sin(np.pi * r)
        new_y = (theta / np.pi) * np.cos(np.pi * r)
        return Point(x=new_x, y=new_y)