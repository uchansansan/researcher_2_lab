import numpy as np
import cv2 as cv

img = cv.imread('c.jpg')

img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

range_low = np.array([0,140,130])
range_up = np.array([60,310,300])
mask = cv.inRange(img_hsv,range_low, range_up)
range_low = np.array([48,16,127])
range_up = np.array([174,85,255])
mask2 = cv.inRange(img_hsv,range_low, range_up)

mask = cv.bitwise_or(mask,mask2)

img_h = cv.Canny(mask,10,200,np.pi/180,3,True)

contours, hierarchy = cv.findContours(img_h, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.imshow('hello',img)

cv.waitKey(0)

cv.imwrite('edited.jpg',img)
cv.destroyAllWindows()
