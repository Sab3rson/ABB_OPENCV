# resizing images

import os
import cv2


img = cv2.imread(os.path.join('.', 'data', 'picture.jpg'))

resized_img = cv2.resize(img, (640, 680))

print(img.shape)
print(resized_img.shape)

cv2.imshow('frame', img)
cv2.imshow('resized image', resized_img)
cv2.waitKey(0)