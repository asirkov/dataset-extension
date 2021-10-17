import argparse
import json
import os

import cv2

import core

ROOT_DIR = os.getcwd()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="relative path to images directory")
ap.add_argument("-a", "--annotations", required=True, help="relative path to annotations file path")

args = vars(ap.parse_args())

# path to images directory
IMAGES_DIR = os.path.join(ROOT_DIR, *os.path.split(args["images"]))
if not os.path.exists(IMAGES_DIR):
    raise Exception("Directory does not exists: ".format(IMAGES_DIR))

# path to annotations directory
ANNOTATIONS_PATH = os.path.join(ROOT_DIR, *os.path.split(args["annotations"]))
if not os.path.exists(ANNOTATIONS_PATH):
    raise Exception("File does not exists: ".format(IMAGES_DIR))


def main():
    annotations = json.load(open(os.path.join(ANNOTATIONS_PATH), "r"))

    for annotation_name, annotation_value in annotations.items():
        file_path = os.path.join(IMAGES_DIR, annotation_value["filename"])
        image = cv2.imread(file_path)

        regions = annotation_value["regions"] if "regions" in annotation_value else []

        cv2.imshow("Image with polygon", core.fill_regions(image.copy(), regions))
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
