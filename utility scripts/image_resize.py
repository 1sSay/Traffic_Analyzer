import cv2 as cv

image = 'C:\\OUR project\\Model_with_tracking\\data\\predicted_photos\\10.JPG'

cv.imwrite("full hd.JPG", cv.resize(cv.imread(image), (1920, 1080)))