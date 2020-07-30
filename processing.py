import numpy as np
import os
import cv2
import imutils

def preprocess_img(img, side=None, crop = True, sensitivity_to_light=50):
    '''
    this function takes a cv image as input, calls the resize function, crops the image to keep only the board, chooses the left / right half of the board or the full board if the child is playing alone, and eventually finds the largest dark shape
    =========

    Parameters : 

    img = OpenCV image
    side = process either left/right side or full frame.  - True by default
    crop = decides if image needs cropping - set crop to False when processing dataset images, they are already cut
    sensitivity_to_light = parameter to turn the background black

    author : @BasCR-hub
    '''

    img = resize(img).copy()

    if crop :
        if side == "left":
            img = img[0:int(img.shape[0]),int(img.shape[1]/2):] # keep only the left half of the board
        elif side == "right" :
            img = img[0:int(img.shape[0]),0:int(img.shape[1]/2)] # keep only the right half of the board
        else:
            img = img[0:-50, 55:-100] # get full frame, if child plays alone  
    
    # get the largest shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # binarize img
    gray[gray>sensitivity_to_light] = 0 # turn background to black
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # ??
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]  # ??
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) # we need the contours to compute Hu moments
    return cnts, img

def resize(img, percent=20):
    '''

    this function takes a cv image as input and resizes it. 
    The primary objective is to make the contouring less sensitive to between-tangram demarcation lines,
    the secondary objective is to speed up processing.

    =========

    Parameters : 

    img : OpenCV image  
    percent : the percentage of the scaling  

    author : @BasCR-hub  
    '''
    scale_percent = percent # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA).copy()
    return img

def display_contour(cnts, img):
    """
    display the contour of the image

    =========

    Parameters : 
    cnts : contours of the forms in the image
    img : OpenCV image

    author : @BasCR-hub
    """
    for c in cnts:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # testing the contour of the image => see with Renata how to include it to integration tests
    img_cv = cv2.imread('data/tangrams/renard.jpg')
    cnts, img = preprocess_img(img_cv, crop=False)
    display_contour(cnts, img)