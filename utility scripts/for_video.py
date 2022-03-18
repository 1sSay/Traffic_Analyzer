import glob
import cv2 as cv

out = cv.VideoWriter("123.avi", cv.VideoWriter_fourcc(*'MJPG'), 30.0, (1920, 1080))

counter = 0

for path in sorted(glob.glob("data\\predicted_photos\\*.JPG"), key=lambda x: int(x.split('\\')[-1][:-4])):
    x, y = 1000, 500

    out.write(cv.imread(path)[y:y + 1080, x:x + 1920])
    counter += 1
    print(counter)

out.release()