import cv2 as cv
import os
import math

from bbox import Bbox
from settings import blob_size, overlapping, object_colors, classes


class Image:
    def __init__(self, image_path, image=None, id=0):
        self.image_path = image_path
        self.image_label = os.path.basename(image_path)
        self.image_id = id

        if image is None:
            self.image = cv.imread(image_path)
        else:
            self.image = image

        self.img_width = image.shape()[0]
        self.img_height = image.shape()[1]

        self.x_count = math.ceil((self.img_width - blob_size) / (blob_size * overlapping))
        self.y_count = math.ceil((self.img_height - blob_size) / (blob_size * overlapping))

        self.x_coords = [i * (self.img_width - blob_size) // self.x_count for i in range(self.x_count)] + [
            self.img_width - blob_size]
        self.y_coords = [i * (self.img_height - blob_size) // self.y_count for i in range(self.y_count)] + [
            self.img_height - blob_size]

        self.bboxes = list()
        self.object_counter = {0: 0,
                               1: 0,
                               2: 0,
                               3: 0}

    def get_label(self):
        return self.image_label

    def get_path(self):
        return self.image_path

    def get_id(self):
        return self.image_id

    def get_image(self):
        return self.image

    def set_bboxes(self, bboxes):
        self.bboxes = bboxes

        for bbox in self.bboxes:
            self.object_counter[bbox.get_label()] += 1
            self.object_counter[3] += 1

    def write(self, writing_path, image_name=None):
        writing_image = self.image

        if image_name is None:
            image_name = self.image_label

        predicts = open(writing_path + ("" if writing_path.endswith("\\") else "\\") +
                        image_name + '.txt')

        for bbox in self.bboxes:
            predicts.write(bbox.get_bbox_for_YOLO())

            cv.rectangle(writing_image,
                         bbox.get_bbox_for_CV2rectangle(),
                         object_colors[bbox.get_label()],
                         1)
            cv.putText(writing_image, f'{classes[bbox.get_label()]} {round(bbox.get_confidence() * 100)}%',
                       (bbox.x_min - 10, bbox.y_min - 10),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       object_colors[bbox.get_label()],
                       2)

            predicts.close()
            cv.imwrite(writing_path + ("" if writing_path.endswith("\\") else "\\") +
                       image_name + '.jpg', writing_image)
