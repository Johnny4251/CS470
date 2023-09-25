
import sys
import numpy as np
import torch
import tensorflow as tf
import cv2
import pandas
import sklearn
from enum import Enum
import math as m

class FilterType(Enum):
    BOX = 0
    GAUSS = 1
    MEDIAN = 2
    LAPLACE = 3
    LAPLACE_SHARP = 4
    SOBEL_X = 5
    SOBEL_Y = 6
    GRAD_MAG = 7
    CUSTOM = 8

"""
Box filtering!
This is a Low-Pass filter
Pretty much the one advantadge to this is
that is really cheap and efficient!

parameters 1-3 are optional.. kernel is optional(custom filter)
"""
def filter(image, filterSize, filterType, kernel=None):
    #output = np.copy(image)
    if filterType == FilterType.BOX:
        output = cv2.boxFilter(image, -1, (filterSize, filterSize))
    elif filterType == FilterType.GAUSS:
        output = cv2.GaussianBlur(image, (filterSize, 1), 0)
    elif filterType == FilterType.MEDIAN:
        output = cv2.medianBlur(image, filterSize)
    elif filterType == FilterType.LAPLACE:
        laplace = cv2.Laplacian(image, cv2.CV_32, 
                               ksize=filterSize, 
                               scale=0.25)
        output = cv2.convertScaleAbs(laplace, alpha=0.5, beta=127.0)
    elif filterType == FilterType.LAPLACE_SHARP:        
        laplace = cv2.Laplacian(image, cv2.CV_32, 
                               ksize=filterSize, 
                               scale=0.25)
        fimage = image.astype("float32")
        fimage -= laplace
        output = cv2.convertScaleAbs(fimage)
    elif filterType == FilterType.SOBEL_X:
        sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, 
                       ksize=filterSize, 
                       scale=.25)
        output = cv2.convertScaleAbs(sx, alpha=0.5, beta=127.0)
    elif filterType == FilterType.SOBEL_Y:
        sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, 
                       ksize=filterSize, 
                       scale=.25)
        output = cv2.convertScaleAbs(sy, alpha=0.5, beta=127.0)
    elif filterType == FilterType.GRAD_MAG:
        sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, 
                       ksize=filterSize, 
                       scale=.25)
        sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, 
                       ksize=filterSize, 
                       scale=.25)
        grad_image = np.absolute(sx) + np.absolute(sy)
        output = cv2.convertScaleAbs(grad_image)
    elif filterType == FilterType.CUSTOM:
        if kernel is None:
            raise ValueError("Cannot use custom filter with None!")
        
        displayScale = np.sum(np.absolute(kernel))
        result = cv2.filter2D(image, cv2.CV_32F, kernel)
        output = cv2.convertScaleAbs(result, 
                                     alpha=1.0/displayScale, 
                                     beta=127.0)
        
    else:
        output = np.copy(image)

    return output

def main():
    if len(sys.argv) <= 1:
        print("Opening webcam...")

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        if not camera.isOpened():
            print("ERROR: Cannot open camera!")
            exit(1)

        windowName = "Webcam"
        cv2.namedWindow(windowName)

        filterSize = 27
        filterType = FilterType.CUSTOM

        kernel = np.zeros((filterSize, filterSize))
        scale = (10*m.pi) / filterSize
        for i in range(filterSize):
            kernel[:,i] = m.sin(i*scale)

        key = -1
        while key == -1:
            _, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output = filter(gray, filterSize,  filterType, kernel)

            cv2.imshow(windowName, frame)
            cv2.imshow("FILTERED", output)

            key = cv2.waitKey(30)

        camera.release()
        cv2.destroyAllWindows()

        print("Closing application...")

    else:
        filename = sys.argv[1]

        print("Loading image:", filename)
        image = cv2.imread(filename) 

        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, image)

        cv2.waitKey(-1)

        cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()