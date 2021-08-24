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
drawing = np.full((img_h.shape[0], img_h.shape[1], 3),0, dtype=np.uint8)
color = (10,100,10)

for i in range(len(contours)):
    cv.drawContours(drawing, contours, i, color, 7, cv.LINE_AA, hierarchy, 0)

drawing = cv.bitwise_and(drawing,drawing, mask=mask)

img_view = np.add(img, drawing)

cv.imshow('hello',img_view)

cv.waitKey(0)

cv.imwrite('edited.jpg',img_view)
cv.destroyAllWindows()
