import cv2
import os

# read video

vid_path = os.path.join('.', 'data', 'web.mp4')

vid = cv2.VideoCapture(vid_path)

# visualize video

ret = True
while ret:
    ret, frame = vid.read()

    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey(80)

vid.release()
cv2.destroyAllWindows()