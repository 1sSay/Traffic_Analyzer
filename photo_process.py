import os
import time
import glob
import math
import concurrent.futures
from sys import stdout

import cv2 as cv
import numpy as np

import onnxruntime as rt

from settings import *
from nms import ExtendedNonMaximumSupression
from bbox import Bbox


def process_image(image, image_path):
    result = []

    image_for_slicing = np.expand_dims(np.transpose(np.array(cv.cvtColor(image, cv.COLOR_BGR2RGB), dtype='float32'), [2, 0, 1]) / 255., axis=0)

    for x in x_coord:
        for y in y_coord:
            sample = image_for_slicing[:, :, y:y + blob_size, x:x + blob_size].copy()

            detections = sess.run(output_names, {input_name: sample})

            for b, s in zip(detections[0][0], detections[1][0]):
                if max(s) > confidence_threshold:
                    class_id = np.argmax(s)
                    confidence = s[class_id]

                    x1, y1, x2, y2 = tuple([int(i * blob_size) for i in b[0]])

                    on_border = x1 < 4 or x2 > blob_size - 4 or y1 < 4 or y2 > blob_size - 4

                    result.append(Bbox(class_id, 0,
                                                  x1 + x, y1 + y, x2 + x, y2 + y,
                                                  confidence=confidence, on_border=on_border))

    bboxes = ExtendedNonMaximumSupression(result, nms_threshold,
                                          destroy_objects_on_border=True,
                                          check_IoA=True)

    count = [0, 0, 0]
    with open(f'{predicted_images}\\' + os.path.basename(image_path)[:-4] + '.txt', 'w') as txt_predicts:
        for bbox in bboxes:
            count[bbox.get_label()] += 1

            txt_predicts.write(bbox.get_bbox_for_YOLO())

            cv.rectangle(image,
                         bbox.get_bbox_for_CV2rectangle(),
                         object_colors[bbox.get_label()],
                         1)
            cv.putText(image, f'{classes[bbox.get_label()]} {round(bbox.get_confidence() * 100)}%',
                       (bbox.x_min - 10, bbox.y_min - 10),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       object_colors[bbox.get_label()],
                       2)

    image[:600, :920, :] //= 4

    cv.putText(image,
               f"Cars: {count[0]}",
               (20, 100), font, color=main_color, fontScale=3, thickness=5)
    cv.putText(image,
               f"Buses: {count[1]}",
               (20, 250), font, color=main_color, fontScale=3, thickness=5)
    cv.putText(image,
               f"Trucks: {count[2]}",
               (20, 400), font, color=main_color, fontScale=3, thickness=5)
    cv.putText(image,
               f"All objects: {count[0] + count[1] + count[2]}",
               (20, 550), font, color=main_color, fontScale=3, thickness=5)

    cv.imwrite(f'{predicted_images}\\{os.path.basename(image_path)}', image)

    stdout.write(f'\r{counter} frames have been predicted')


if __name__ == '__main__':
    sess = rt.InferenceSession(model_without_embeddings)

    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name

    total_start = time.time()  # Для вычисления времени работы

    img_width = image_process_width
    img_height = image_process_height
    counter = 0  # frame counter

    x_count = math.ceil((img_width - blob_size) / (blob_size * overlapping))
    y_count = math.ceil((img_height - blob_size) / (blob_size * overlapping))

    x_coord = [i * (img_width - blob_size) // x_count for i in range(x_count)] + [img_width - blob_size]
    y_coord = [i * (img_height - blob_size) // y_count for i in range(y_count)] + [img_height - blob_size]

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=4) as executor:
        for i in executor.map(lambda x: process_image(cv.imread(x), x),
                              [image_path for image_path in glob.glob(image_folder + '\\*') if
                               image_path.endswith((".jpg", ".JPG", '.png', '.PNG'))]):
            counter += 1

    total_finish = time.time()

    print('',  # Вывод основной информации
          f'Total processing time: {(total_finish - total_start):.2f} sec',
          f'Frames found: {counter}',
          f'Average processing time: {(total_finish - total_start) / counter:.2f} sec/img',
          f'Processing speed: {counter / (total_finish - total_start):.2f} img/sec',
          sep='\n')
