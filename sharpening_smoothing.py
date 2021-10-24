import cv2 as cv

# read img file
img = cv.imread('photos/dog.jpg')
cv.imshow('Img', img)


# # Blur (smooth) the img
# smooth = cv.GaussianBlur(img,(5,5), cv.BORDER_DEFAULT)
# cv.imshow('Gaussian Blur', smooth)

# # Averaging blur
# average = cv.blur(img, (5,5))
# cv.imshow('Average Blur', average)

# # Median Blur
# median = cv.medianBlur(img, 5)
# cv.imshow('Median Blur', median)

# # Bilateral Blur -> edges remains without distortion
# bilateral = cv.bilateralFilter(img, 5, 15, 15)
# cv.imshow('Bilateral', bilateral)

# # wait any key to exit
# cv.waitKey(0)

import numpy as np

# convert to grey scale
grey_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow('Grey Img', grey_img)

# use canny edge detector
canny = cv.Canny(grey_img, 125, 175)
cv.imshow('Canny Img', canny)

# Laplacian edge detector
lap = cv.Laplacian(grey_img, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel edge detector
sobelx = cv.Sobel(grey_img, cv.CV_64F, 1, 0)
sobely = cv.Sobel(grey_img, cv.CV_64F, 0, 1)
cv.imshow('Sobelx', sobelx)
cv.imshow('Sobely', sobely)

# Combine the 2 axis
combined_sobel = cv.bitwise_or(sobelx,sobely)
cv.imshow('Combined Sobel', combined_sobel)

# # wait any key to exit
cv.waitKey(0)