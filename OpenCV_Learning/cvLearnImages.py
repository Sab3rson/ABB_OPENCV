import cv2
import os

# read image
image_path = os.path.join('.','data', 'picture.jpg')

img = cv2.imread(image_path)

# write image

cv2.imwrite(os.path.join('.', 'data', 'picture_out.jpg'), img)

# Visualize image

cv2.imshow('image frame', img)
cv2.waitKey(200)

