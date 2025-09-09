import os
import cv2

img = cv2.imread(os.path.join('.','data','picture.jpg'))



k_size = 11
img_blur = cv2.blur(img, (k_size, k_size))
img_gaussian_blur = cv2.GaussianBlur(img, (k_size, k_size), 3)
img_median_blur = cv2.medianBlur(img,k_size)

cv2.imshow('img', img)
cv2.imshow('img2', img_blur)
cv2.imshow('g_img', img_gaussian_blur)
cv2.imshow('m_img', img_median_blur)
cv2.waitKey(0) 