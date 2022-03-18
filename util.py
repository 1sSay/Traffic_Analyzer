import math


def calculate_distance(point1, point2):
    return math.sqrt(abs(point1[0] - point2[0]) ** 2 + abs(point1[1] - point2[1]) ** 2)
