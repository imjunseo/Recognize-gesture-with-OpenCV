"""Small, controlled hand poses in the MediaPipe landmark order."""

from math import cos, radians, sin
from types import SimpleNamespace


POSES = {
    "Open palm": (1, 1, 1, 1, 1),
    "Victory": (0, 1, 1, 0, 0),
    "Middle finger": (0, 0, 1, 0, 0),
    "Pinky": (0, 0, 0, 0, 1),
    "Thumb": (1, 0, 0, 0, 0),
    "Pointing up": (0, 1, 0, 0, 0),
    "Closed fist": (0, 0, 0, 0, 0),
}
EDGES = (
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    *((i, i + 1) for i in range(1, 20) if i % 4 != 0),
)


def hand_pose(extended, angle=0):
    points = [(0, 0, 0)]
    if extended[0]:
        points.extend([(-0.3, 0.28, 0), (-0.52, 0.43, 0),
                       (-0.73, 0.57, 0), (-0.93, 0.71, 0)])
    else:
        points.extend([(-0.3, 0.28, 0), (-0.45, 0.45, 0),
                       (-0.38, 0.54, 0.22), (-0.19, 0.54, 0.23)])
    for x, length, straight in zip(
        (-0.36, -0.11, 0.15, 0.41), (0.66, 0.77, 0.69, 0.54), extended[1:]
    ):
        if straight:
            points.extend((x, 0.63 + length * step, 0) for step in (0, 0.42, 0.75, 1))
        else:
            points.extend([(x, 0.63, 0), (x, 0.85, 0.03),
                           (x, 0.81, 0.23), (x, 0.64, 0.31)])
    c, s = cos(radians(angle)), sin(radians(angle))
    return [SimpleNamespace(x=c * x + s * z, y=y, z=-s * x + c * z) for x, y, z in points]
