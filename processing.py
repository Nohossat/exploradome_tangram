import numpy as np
import os
import cv2
import imutils

def get_files():
    """
    get images in jpg format in tangrams folder
    """
    images = {}
    dirname = os.getcwd() + '/data/tangrams'
    assert os.path.exists(dirname), "the directory doesn't exist"

    for file in os.listdir(dirname):
        filename, file_extension = os.path.splitext(file) # we just want the filename to save the path
        if file.endswith((".jpg", ".JPG")):
            images[filename] = os.path.join(dirname, file)
    return images

def preprocess_img_test(img, sensitivity_to_light=50):
    '''
    this function takes a cv image as input, calls the resize function, crops the image to keep only the board, 
    chooses the left or right half of the board, based on user input and eventually finds the largest dark shape
    '''

    # resize operations
    img = resize(img, 20).copy()

    # get the largest shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # binarize img
    gray[gray>sensitivity_to_light] = 0 # turn background to black
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1] # threshold ???
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts, img

def preprocess_img(img, left_side=True, sensitivity_to_light=50):
    '''
    this function takes a cv image as input, calls the resize function, crops the image to keep only the board, 
    chooses the left or right half of the board, based on user input and eventually finds the largest dark shape
    '''

    # resize operations
    img = resize(img,20)[0:-50, 55:-100].copy() # resize img to 20% and crop to keep only board

    if left_side:
        img = img[0:int(img.shape[0]),0:int(img.shape[1]/2)] # keep only the left half of the board
    else:
        img = img[0:int(img.shape[0]),int(img.shape[1]/2):] # keep only the right half of the board
    
    # get the largest shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # binarize img
    gray[gray>sensitivity_to_light] = 0 # turn background to black
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1] # threshold ???
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts, img

def resize(img, percent=20):
    '''
    this function takes a cv image as input and resizes it. 
    The primary objective is to make the contouring less sensitive to between-tangram demarcation lines,
    the secondary objective is to speed up processing.
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
    """
    for c in cnts:
        #compute the center of the contour
        M = cv2.moments(c)
        huM = cv2.HuMoments(M)
        # draw the contour and center of the shape on the image
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    # show the image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
