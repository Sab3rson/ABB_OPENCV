import os
 
import cv2


img = cv2.imread(os.path.join('.', 'data', 'picture.jpg'))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('img_rgb', img_rgb) 
cv2.imshow('img_gray', img_bw)
cv2.imshow('img_gray', img_HSV)
cv2.imshow('img', img)
cv2.waitKey(0)