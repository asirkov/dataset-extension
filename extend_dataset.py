import argparse
import json
import os
import sys

import cv2

import core

ROOT_DIR = os.getcwd()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="relative path to images directory")
ap.add_argument("-a", "--annotations", required=True, help="relative path to annotations directory")
ap.add_argument("-v", "--verbose", required=False, action='store_true')

args = vars(ap.parse_args())

# path to images directory
IMAGES_DIR = os.path.join(ROOT_DIR, *os.path.split(args["images"]))
if not os.path.exists(IMAGES_DIR):
    raise Exception("Directory does not exists: ".format(IMAGES_DIR))

# path to annotations directory
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, *os.path.split(args["annotations"]))
if not os.path.exists(ANNOTATIONS_DIR):
    raise Exception("Directory does not exists: ".format(IMAGES_DIR))

VERBOSE = bool(args["verbose"]) if "verbose" in args else False


def get_image_flipped_horizontal(image, annotation):
    filename = annotation["filename"]
    size = annotation["size"]
    file_attributes = annotation["file_attributes"]
    regions = annotation["regions"] if "regions" in annotation else []

    new_image = cv2.flip(image, flipCode=1)

    regions_fh = core.flip_regions_horizontal(regions, weight=image.shape[1])
    filename_fh = filename.replace(".", "_fh.")

    new_annotation = {
        "filename": filename_fh,
        "size": size,
        "regions": regions_fh,
        "file_attributes": file_attributes,
    }
    return new_image, new_annotation


def get_image_flipped_vertical(image, annotation):
    filename = annotation["filename"]
    size = annotation["size"]
    file_attributes = annotation["file_attributes"]
    regions = annotation["regions"] if "regions" in annotation else []

    new_image = cv2.flip(image, flipCode=0)

    regions_fv = core.flip_regions_vertical(regions, height=image.shape[0])
    filename_fv = filename.replace(".", "_fv.")

    new_annotation = {
        "filename": filename_fv,
        "size": size,
        "regions": regions_fv,
        "file_attributes": file_attributes,
    }
    return new_image, new_annotation


def get_image_grayscale(image, annotation):
    filename = annotation["filename"]
    size = annotation["size"]
    file_attributes = annotation["file_attributes"]
    regions = annotation["regions"] if "regions" in annotation else []

    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filename_gs = filename.replace(".", "_gs.")

    new_annotation = {
        "filename": filename_gs,
        "size": size,
        "regions": regions,
        "file_attributes": file_attributes,
    }
    return new_image, new_annotation


def get_image_gaussian_blured(image, annotation):
    filename = annotation["filename"]
    size = annotation["size"]
    file_attributes = annotation["file_attributes"]
    regions = annotation["regions"] if "regions" in annotation else []

    new_image = cv2.GaussianBlur(image, ksize=(31, 31), sigmaX=0, sigmaY=0)

    filename_gb = filename.replace(".", "_gb.")

    new_annotation = {
        "filename": filename_gb,
        "size": size,
        "regions": regions,
        "file_attributes": file_attributes,
    }
    return new_image, new_annotation


def get_image_rotated(image, annotation, angle, filename_suffix):
    filename = annotation["filename"]
    size = annotation["size"]
    file_attributes = annotation["file_attributes"]
    regions = annotation["regions"] if "regions" in annotation else []

    rotated_image = core.rotate_image(image, -angle)

    center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
    rotated_center = (int(rotated_image.shape[1] / 2), int(rotated_image.shape[0] / 2))
    shift_x, shift_y = rotated_center[0] - center[0], rotated_center[1] - center[1]

    regions_r = core.rotate_regions(regions, center, shift_x, shift_y, angle)
    filename_r = filename.replace(".", "{}.".format(filename_suffix))

    new_annotation = {
        "filename": filename_r,
        "size": size,
        "regions": regions_r,
        "file_attributes": file_attributes,
    }
    return rotated_image, new_annotation


def get_image_rotated15(image, annotation):
    return get_image_rotated(image, annotation, 15, "_r15")


def get_image_rotated30(image, annotation):
    return get_image_rotated(image, annotation, 30, "_r30")


def get_image_rotated90(image, annotation):
    return get_image_rotated(image, annotation, 90, "_r90")


def get_image_rotated180(image, annotation):
    return get_image_rotated(image, annotation, 180, "_r180")


def main():
    annotations = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotations.json"), "r"))
    new_annotations = annotations.copy()

    for annotation_name, annotation in annotations.items():
        filename = annotation["filename"]
        regions = annotation["regions"] if "regions" in annotation else []

        file_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(file_path):
            print("Failed reading image, path: {}".format(file_path), file=sys.stderr)
            continue

        image = cv2.imread(file_path)

        if VERBOSE:
            cv2.imshow("Image", core.fill_regions(image.copy(), regions))
            cv2.waitKey(0)

        annotation_fh_name = annotation_name + "_fh"
        if annotation_fh_name not in annotations:
            image_fh, annotation_fh = get_image_flipped_horizontal(image.copy(), annotation.copy())

            if VERBOSE:
                cv2.imshow("Image", core.fill_regions(image_fh.copy(), annotation_fh["regions"]))
                cv2.waitKey(0)

            image_path = os.path.join(IMAGES_DIR, annotation_fh["filename"])
            if cv2.imwrite(image_path, image_fh):
                print("Image saved, path: {}".format(image_path))
            else:
                raise Exception("Failed to write image, path: {}".format(annotation_fh["filename"]))

            new_annotations[annotation_fh_name] = annotation_fh

        annotation_fv_name = annotation_name + "_fv"
        if annotation_fv_name not in annotations:
            image_fv, annotation_fv = get_image_flipped_vertical(image.copy(), annotation.copy())

            if VERBOSE:
                cv2.imshow("Image", core.fill_regions(image_fv.copy(), annotation_fv["regions"]))
                cv2.waitKey(0)

            image_path = os.path.join(IMAGES_DIR, annotation_fv["filename"])
            if cv2.imwrite(image_path, image_fv):
                print("Image saved, path: {}".format(image_path))
            else:
                raise Exception("Failed to write image, path: {}".format(annotation_fv["filename"]))

            new_annotations[annotation_fv_name] = annotation_fv

        annotation_gs_name = annotation_name + "_gs"
        if annotation_gs_name not in annotations:
            image_gs, annotation_gs = get_image_grayscale(image.copy(), annotation.copy())

            if VERBOSE:
                cv2.imshow("Image", core.fill_regions(image_gs.copy(), annotation_gs["regions"]))
                cv2.waitKey(0)

            image_path = os.path.join(IMAGES_DIR, annotation_gs["filename"])
            if cv2.imwrite(image_path, image_gs):
                print("Image saved, path: {}".format(image_path))
            else:
                raise Exception("Failed to write image, path: {}".format(annotation_gs["filename"]))

            new_annotations[annotation_gs_name] = annotation_gs

        annotation_gb_name = annotation_name + "_gb"
        if annotation_gb_name not in annotations:
            image_gb, annotation_gb = get_image_gaussian_blured(image.copy(), annotation.copy())

            if VERBOSE:
                cv2.imshow("Image", core.fill_regions(image_gb.copy(), annotation_gb["regions"]))
                cv2.waitKey(0)

            image_path = os.path.join(IMAGES_DIR, annotation_gb["filename"])
            if cv2.imwrite(image_path, image_gb):
                print("Image saved, path: {}".format(image_path))
            else:
                raise Exception("Failed to write image, path: {}".format(annotation_gb["filename"]))

            new_annotations[annotation_gb_name] = annotation_gb

        annotation_r15_name = annotation_name + "_r15"
        if annotation_r15_name not in annotations:
            image_r15, annotation_r15 = get_image_rotated15(image.copy(), annotation.copy())

            if VERBOSE:
                cv2.imshow("Image", core.fill_regions(image_r15.copy(), annotation_r15["regions"]))
                cv2.waitKey(0)

            image_path = os.path.join(IMAGES_DIR, annotation_r15["filename"])
            if cv2.imwrite(image_path, image_r15):
                print("Image saved, path: {}".format(image_path))
            else:
                raise Exception("Failed to write image, path: {}".format(annotation_r15["filename"]))

            new_annotations[annotation_r15_name] = annotation_r15

        annotation_r30_name = annotation_name + "_r30"
        if annotation_r30_name not in annotations:
            image_r30, annotation_r30 = get_image_rotated30(image.copy(), annotation.copy())

            if VERBOSE:
                cv2.imshow("Image", core.fill_regions(image_r30.copy(), annotation_r30["regions"]))
                cv2.waitKey(0)

            image_path = os.path.join(IMAGES_DIR, annotation_r30["filename"])
            if cv2.imwrite(image_path, image_r30):
                print("Image saved, path: {}".format(image_path))
            else:
                raise Exception("Failed to write image, path: {}".format(annotation_r30["filename"]))

            new_annotations[annotation_r30_name] = annotation_r30

        annotation_r90_name = annotation_name + "_r90"
        if annotation_r90_name not in annotations:
            image_r90, annotation_r90 = get_image_rotated90(image.copy(), annotation.copy())

            if VERBOSE:
                cv2.imshow("Image", core.fill_regions(image_r90.copy(), annotation_r90["regions"]))
                cv2.waitKey(0)

            if not cv2.imwrite(os.path.join(IMAGES_DIR, annotation_r90["filename"]), image_r90):
                raise Exception("Failed to write image, path: {}".format(annotation_r90["filename"]))

            new_annotations[annotation_r90_name] = annotation_r90

        annotation_r180_name = annotation_name + "_r180"
        if annotation_r180_name not in annotations:
            image_r180, annotation_r180 = get_image_rotated180(image.copy(), annotation.copy())

            if VERBOSE:
                cv2.imshow("Image", core.fill_regions(image_r180.copy(), annotation_r180["regions"]))
                cv2.waitKey(0)

            image_path = os.path.join(IMAGES_DIR, annotation_r180["filename"])
            if cv2.imwrite(image_path, image_r180):
                print("Image saved, path: {}".format(image_path))
            else:
                raise Exception("Failed to write image, path: {}".format(annotation_r180["filename"]))

            new_annotations[annotation_r180_name] = annotation_r180

    json.dump(new_annotations, open(os.path.join(ANNOTATIONS_DIR, "new_annotations.json"), "w"))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
