import colorsys
import math
import random

import cv2
import numpy as np

from scipy import ndimage


def random_colors(count, bright=True):
    brightness = 1.0 if bright else 0.7
    colors = list(map(lambda color: [int(c * 255) for c in colorsys.hsv_to_rgb(*color)],
                      [(i / count, 1, brightness) for i in range(count)]))
    random.shuffle(colors)
    return colors


def fill_regions(image, regions, alpha=0.7):
    colors = random_colors(len(regions))
    shapes = np.zeros_like(image, np.uint8)

    for i, region in enumerate(regions):
        shape_attributes = region["shape_attributes"]
        pts = np.column_stack((shape_attributes["all_points_x"], shape_attributes["all_points_y"]))
        cv2.fillPoly(shapes, [pts], colors[i])

    mask = shapes.astype(bool)
    image[mask] = cv2.addWeighted(src1=image, alpha=1 - alpha, src2=shapes, beta=alpha, gamma=0)[mask]

    return image


def flip_regions_horizontal(regions, weight):
    new_regions = []

    for region in regions:
        new_region = {
            "region_attributes": region["region_attributes"],
            "shape_attributes": {
                "name": region["shape_attributes"]["name"],
                "all_points_x": [weight - x for x in region["shape_attributes"]["all_points_x"]],
                "all_points_y": region["shape_attributes"]["all_points_y"]
            }
        }

        new_regions.append(new_region)

    return new_regions


def flip_regions_vertical(regions, height):
    new_regions = []

    for region in regions:
        new_region = {
            "region_attributes": region["region_attributes"],
            "shape_attributes": {
                "name": region["shape_attributes"]["name"],
                "all_points_x": region["shape_attributes"]["all_points_x"],
                "all_points_y": [height - y for y in region["shape_attributes"]["all_points_y"]],
            }
        }

        new_regions.append(new_region)

    return new_regions


def rotate_image(image, angle):
    return ndimage.rotate(image, angle)


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def rotate_regions(regions, center, shift_weight, shift_height, angle):
    new_regions = []

    for region in regions:
        shape_attributes = region["shape_attributes"]

        points = np.column_stack((shape_attributes["all_points_x"], shape_attributes["all_points_y"]))
        rotated_points = []

        for point in points:
            rotated_point = rotate_point(center, point, math.radians(angle))
            rotated_points.append((int(rotated_point[0]), int(rotated_point[1])))

        new_region = {
            "region_attributes": region["region_attributes"],
            "shape_attributes": {
                "name": region["shape_attributes"]["name"],
                "all_points_x": [shift_weight + x for x, y in rotated_points],
                "all_points_y": [shift_height + y for x, y in rotated_points],
            }
        }

        new_regions.append(new_region)

    return new_regions
