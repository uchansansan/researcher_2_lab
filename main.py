import numpy as np
import cv2 as cv

img = cv.imread('c.jpg')
img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

range_low = np.array([0,150,140])
range_up = np.array([50,300,290])

mask = cv.inRange(img_hsv,range_low, range_up)
img_h = cv.bitwise_and(img,img,mask=mask)

cv.imshow('hello',img_h)

cv.waitKey(0)
cv.imwrite('edited.jpg',img_hsv)
cv.destroyAllWindows()
