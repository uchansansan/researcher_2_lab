import numpy as NP
import cv2 as OpenCV
from time import sleep as Delay

import numpy as np


def reverseBlock(x, y, matrix):
    matrix[y][x] = 1 if matrix[y][x] == 0 else 0

def isBlockLife(x,y,matrix):
    return bool(matrix[y][x])

def neighboursCount(x,y,matrix):
    amount = 0
    for i in range(-1,2):
        for n in range(-1,2):
            if not (i or n):
                continue
            elif isBlockLife((x+n)%64, (y+i)%64, matrix):
                amount += 1
                print(n,i)
    return amount

def drawMatrix(matrix):
    return NP.array([[Types[0] if el == 0 else Types[1] for el in n] for n in matrix], dtype = NP.uint8)

def updateScreen(matrix,text=''):
    resized = OpenCV.resize(drawMatrix(Matrix), (0,0),fx=900/64,fy=900/64, interpolation=OpenCV.INTER_AREA)
    if text != '':
        resized = OpenCV.putText(resized,text,(50,50),OpenCV.FONT_HERSHEY_SIMPLEX,1,(200,200,200),2)
    OpenCV.imshow('gml', resized)

def Tick(matrix,tick):
    matrixCopy = matrix.copy()
    lifes = 0
    for i in range(64):
        for n in range(64):
            neighbours = neighboursCount(n,i,matrixCopy)
            if isBlockLife(n,i,matrixCopy):
                print('x:',n,'y:',i,'neighbours:',neighbours)
                print('life')
                if neighbours > 3 or neighbours < 2:
                    reverseBlock(n,i,matrix)
                    print(matrix[i][n])
                    print('is died')
                else:
                    lifes += 1
                    print('is live')
            else:
                if neighbours == 3:
                    reverseBlock(n%64,i%64,matrix)
    print(lifes)
    if lifes == 0 or np.array_equal(matrix, matrixCopy):
        return 'no'
    tick = 'Generation '+str(tick)
    updateScreen(matrix,tick)
    return 0


Types = {
    0: [30, 30, 30],
    1: [120, 191, 231]
}

Matrix = NP.full((64,64),0)

def mouse (event, x, y, flags, n):
    pos = [int(x//(900/64)),int(y//(900/64))]
    if event == OpenCV.EVENT_LBUTTONDOWN:
        print(pos)
        reverseBlock(*pos, Matrix)
    elif event == OpenCV.EVENT_RBUTTONDOWN:
        print(neighboursCount(*pos, Matrix))

#64x64blocks
OpenCV.namedWindow('gml')
OpenCV.setMouseCallback('gml', mouse)

while True:
    key = OpenCV.waitKey(1)

    if key == ord('n'):
        break
    updateScreen(Matrix)

tacktes = 1
while True:
    key = OpenCV.waitKey(1)

    if key == ord('q'):
        break
    tacktes += 1
    ret = Tick(Matrix,tacktes)
    if ret == 'no':
        updateScreen(Matrix,'You lose')
        Delay(5)
        break

OpenCV.destroyAllWindows()
