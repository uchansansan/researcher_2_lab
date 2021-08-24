import numpy as np
import cv2 as cv
from cv2 import aruco


class VideoDetection:
    def __init__(self, visualization=False):
        self._SQUARE_BOUND = 3000
        self._DELAY = 50
        self._RED_LOW = np.array([150, 150, 255])
        self._RED_HIGH = np.array([13, 16, 153])
        self._VIZUALIZATION = visualization 

        self._is_scanned = False
        self._timer = 0

        self._core = np.ones([3,3])

    def scan(self, img, tick):
        frame_markers = img
        ids = None
        lines = self._scan_lines(img)

        if self._is_scanned and tick-self._timer >= self._DELAY:
            self._is_scanned = False
            self._timer = tick
            return ids, frame_markers
            
        if lines is not None and len(lines) >= 2: 
            corners, ids = self._scan_code(img)
            square = self._GaussSquare(corners)

            if square  < self._SQUARE_BOUND:
                ids = None

            if self._is_scanned:
                ids = None
                
            if ids is not None:
                if not self._is_scanned:
                    self._timer = tick
                    self._is_scanned = True

            if self._VIZUALIZATION:
                if ids is not None:
                    frame_markers = self._draw_code_box(corners, ids, frame_markers)
                frame_markers = self._draw_lines(lines, frame_markers)

        cv.rectangle(frame_markers, (self.x-self.diff_x, self.y ), (self.diff_x,self.diff_y),(255,0,0), thickness=2)
        return ids, frame_markers

    def _draw_lines(self, lines, image):
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(image,(x1+self.diff_x,y1+self.diff_y),(x2+self.diff_x,y2+self.diff_y),(0,255,0),2)
        return image

    def _draw_code_box(self, corners, ids, image):
        frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
        return frame_markers

    def _scan_lines(self, img):
        mask = self._split_mask(self._sdelat_krasivo(cv.inRange(img, self._RED_HIGH, self._RED_LOW)))
        edges = cv.Canny(mask, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=150)
        return lines

    def _scan_code(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        return corners, ids

    def _GaussSquare(self, vertex):
        right, left, res = 0,0,0
        if vertex != []:
            for i in range(len(vertex[0][0])):
                if i+1 < len(vertex[0][0]):
                    right += vertex[0][0][i][0] * vertex[0][0][i+1][1]
                    left += vertex[0][0][i][1] * vertex[0][0][i+1][0]
            res = right - left
        return abs(res)

    def _split_mask(self, mask):
        y,x = mask.shape
        diff_y, diff_x = round(y/1.4), round(x/7) 
        self.y, self.x, self.diff_x, self.diff_y = y,x, diff_x, diff_y
        cropped_mask = mask.copy()[ diff_y:, diff_x:-diff_x]
        return cropped_mask

    def _sdelat_krasivo(self, mask):
        er = cv.erode(mask, self._core, iterations=2)
        dil = cv.dilate(er, self._core, iterations=4)
        return dil
