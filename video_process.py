import concurrent.futures
import time

import cv2 as cv
import numpy as np
import onnxruntime as rt
from sys import stdout

from settings import *
from bbox import Bbox
from nms import ExtendedNonMaximumSupression
from util import *


def process_image(image, frame_id):
    result = []

    image_for_slicing = np.expand_dims(np.transpose(np.array(cv.cvtColor(image, cv.COLOR_BGR2RGB), dtype='float32'), [2, 0, 1]) / 255., axis=0)

    for x in x_coord:
        for y in y_coord:
            sample = image_for_slicing[:, :, y:y + blob_size, x:x + blob_size].copy()

            detections = sess.run(output_names, {input_name: sample})
            sample_detections = list()

            for b, s in zip(detections[0][0], detections[1][0]):
                if max(s) > confidence_threshold:
                    class_id = np.argmax(s)
                    confidence = s[class_id]

                    x1, y1, x2, y2 = tuple([int(i * blob_size) for i in b[0]])

                    on_border = x1 < 4 or x2 > blob_size - 4 or y1 < 4 or y2 > blob_size - 4

                    sample_detections.append(Bbox(class_id, frame_id,
                                                  x1 + x, y1 + y, x2 + x, y2 + y,
                                                  confidence=confidence, on_border=on_border))

                    for bbox in sample_detections:
                        result.append(bbox)

    return ExtendedNonMaximumSupression(result, nms_threshold,
                                        destroy_objects_on_border=True,
                                        destroy_low_confidence_objects=True,
                                        check_IoA=True)


if __name__ == '__main__':
    # process model
    sess = rt.InferenceSession(model_without_embeddings)

    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name

    total_start = time.time()  # Для вычисления времени работы

    # read video
    cap = cv.VideoCapture(video_name)

    img_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    counter = 0  # frame counter
    object_found = 0

    x_count = math.ceil((img_width - blob_size) / (blob_size * overlapping))
    y_count = math.ceil((img_height - blob_size) / (blob_size * overlapping))

    x_coord = [i * (img_width - blob_size) // x_count for i in range(x_count)] + [img_width - blob_size]
    y_coord = [i * (img_height - blob_size) // y_count for i in range(y_count)] + [img_height - blob_size]

    if cap.isOpened():
        stdout.write(f"\n{video_name} found\n")
    else:
        print("Error while opening...")
        exit(1)

    out = cv.VideoWriter(predicted_video, cv.VideoWriter_fourcc(*'MJPG'), 30.0, (img_width, img_height))

    ret = True

    bboxes_from_previous_frame = []

    while cap.isOpened() and ret:
        ret, frame = cap.read()
        if ret:
            bboxes = process_image(frame, counter)

            for bbox in bboxes:
                for bbox_prev in bboxes_from_previous_frame:
                    if bbox.get_label() != bbox_prev.get_label():
                        continue

                    distance = calculate_distance(bbox.get_centre(), bbox_prev.get_centre())

                    if distance < 40:
                        bbox.id = bbox_prev.id
                        bbox.prev_centre = bbox_prev.get_centre()
                        bbox.velocity = calculate_distance(bbox.get_centre(), bbox_prev.get_centre()) * 3.6 * 30 / 25

                if bbox.id == -1:
                    bbox.id = object_found
                    object_found += 1

            count = [0, 0, 0]
            for bbox in bboxes:
                cv.rectangle(frame,
                             bbox.get_bbox_for_CV2rectangle(),
                             object_colors[bbox.get_label()],
                             2)

                # if bbox.velocity > 10:
                #     cv.arrowedLine(frame, bbox.get_1st_point_for_point(50), bbox.get_2nd_point_for_arrow(150), object_colors[bbox.get_label()], thickness=2, tipLength=0.5)

                cv.putText(frame, f'{bbox.id} {classes[bbox.get_label()]} {round(bbox.get_confidence() * 100)}% {round(bbox.velocity)} km/h',
                           (bbox.x_min - 10, bbox.y_min - 10),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.75,
                           object_colors[bbox.get_label()],
                           2)

                count[bbox.get_label()] += 1

            frame[:750, :920, :] //= 4

            cv.putText(frame,
                       f"Cars: {count[0]}",
                       (20, 100), font, color=main_color, fontScale=3, thickness=5)
            cv.putText(frame,
                       f"Buses: {count[1]}",
                       (20, 250), font, color=main_color, fontScale=3, thickness=5)
            cv.putText(frame,
                       f"Trucks: {count[2]}",
                       (20, 400), font, color=main_color, fontScale=3, thickness=5)
            cv.putText(frame,
                       f"All objects: {count[0] + count[1] + count[2]}",
                       (20, 550), font, color=main_color, fontScale=3, thickness=5)
            cv.putText(frame,
                       f"Objects found: {object_found}",
                       (20, 700), font, color=main_color, fontScale=3, thickness=5)

            out.write(frame)

            cv.imwrite(f'{predicted_images}\\{counter}.JPG', frame)

            stdout.write(f'\r{counter} frames have been predicted')

            bboxes_from_previous_frame = bboxes.copy()

            counter += 1

    cap.release()
    out.release()

    total_finish = time.time()

    print('',  # Вывод основной информации
          f'Total processing time: {(total_finish - total_start):.2f} sec',
          f'Frames found: {counter}',
          f'Average processing time: {(total_finish - total_start) / counter:.2f} sec/img',
          f'Processing speed: {counter / (total_finish - total_start):.2f} img/sec',
          sep='\n')
