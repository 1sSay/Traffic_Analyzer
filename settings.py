import cv2 as cv

# For video testing
video_name = 'data\\videos_for_predicting\\DJI_0004.MP4'
predicted_video = 'data\\predicted_videos\\predicted.avi'

# Predicted
image_folder = 'data\\photos_for_predicting\\'
predicted_images = 'data\\predicted_photos\\'
true_bboxes = ''

dataset_path = 'C:\\OUR project\\model_testing\\probnii'
name_for_stats = 'Evening'

IoU_threshold_TP = 0.75
IoU_threshold_FP = 0.10

# Frame
blob_size = 832
overlapping = 0.5

image_process_width = 3840
image_process_height = 2160



model_with_embeddings = 'YOLOv4_with_embeddings.onnx'
model_without_embeddings = 'YOLOv4_without_embeddings.onnx'

# Thresholds
confidence_threshold = 0.5
nms_threshold = 0.3
width_and_height_threshold = 40
area_threshold = {
    0: 1200,
    1: 1600,
    2: 1600
}

# Objects
classes = {
    0: "car",
    1: "bus",
    2: "truck"
}

font = cv.QT_FONT_NORMAL
object_colors = {
    0: (96, 0, 255),
    1: (64, 255, 0),
    2: (255, 96, 0)
}
main_color = (255, 255, 255)
