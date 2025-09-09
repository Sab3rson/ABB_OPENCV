#image cropping

import os

import cv2

img = cv2.imread(os.path.join('.','data','picture.jpg'))

print(img.shape)

cropped_img = img[100:640, 200:2000]

cv2.imshow('img', img)
cv2.imshow('croppedimg', cropped_img)
cv2.waitKey(0)