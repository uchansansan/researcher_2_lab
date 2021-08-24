import numpy as np
import cv2 as cv

img = cv.imread('c.jpg')

img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

def Track(values):
    range_low = np.array([0,140,130])
    range_up = np.array([60,310,300])
    mask = cv.inRange(img_hsv,range_low, range_up)
    range_low = np.array([*values[0:3]])
    range_up = np.array([*values[3:6]])
    print(range_low,range_up)
    mask2 = cv.inRange(img_hsv,range_low, range_up)

    mask = cv.bitwise_or(mask,mask2)

    img_h = cv.Canny(mask,10,200,np.pi/180,3,True)

    contours, hierarchy = cv.findContours(img_h, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    drawing = np.full((img_h.shape[0], img_h.shape[1], 3),0, dtype=np.uint8)
    color = (10,100,10)

    for i in range(len(contours)):
        cv.drawContours(drawing, contours, i, color, 3, cv.LINE_AA, hierarchy, 0)

    drawing = cv.bitwise_and(drawing,drawing, mask=mask)

    img_view = np.add(img, drawing)

    cv.imshow('hello',img_view)
    cv.imshow('hello2', cv.resize(mask2,None,None,0.5,0.5))


cv.namedWindow('hello')
cv.namedWindow('hello2')

cv.createTrackbar('1l', 'hello2', 0, 360,lambda a:None)
cv.createTrackbar('2l', 'hello2', 0, 255,lambda a: None)
cv.createTrackbar('3l','hello2',0,255,lambda a: None)

cv.createTrackbar('1h', 'hello2', 0, 360,lambda a: None)
cv.createTrackbar('2h', 'hello2', 0, 255,lambda a: None)
cv.createTrackbar('3h', 'hello2', 0, 255,lambda a: None)

while True:
    h1 = cv.getTrackbarPos('1h','hello2')
    h2 = cv.getTrackbarPos('2h','hello2')
    h3 = cv.getTrackbarPos('3h','hello2')
    l1 = cv.getTrackbarPos('1l','hello2')
    l2 = cv.getTrackbarPos('2l','hello2')
    l3 = cv.getTrackbarPos('3l','hello2')

    Track((l1,l2,l3,h1,h2,h3))

    k = cv.waitKey(1)
    if k == 27:
        break
cv.imwrite('edited.jpg',img_hsv)
cv.destroyAllWindows()
