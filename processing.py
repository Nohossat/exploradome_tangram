import numpy as np
import os
import cv2
import imutils

def preprocess_img(img, side=None, sensitivity_to_light=50):
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

    img = resize(img,side).copy()
    image_blurred = blur(img,3)
    cnts = get_contours(image_blurred)
    image_triangles_squares = extract_triangles_squares(cnts, img)
    

    blurred_triangles_squared = blur(image_triangles_squares,7,sensitivity_to_light='ignore').copy()
    final_cnts = get_contours(blurred_triangles_squared)
    return final_cnts

def extract_triangles_squares(cnts, image):    
    cnts_output = []
    out_image = np.zeros(image.shape, image.dtype)
    for idx,cnt in enumerate(cnts):
        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetre, True)

        area = cv2.contourArea(cnt)
        img_area = image.shape[0] * image.shape[1]
        
        if area/img_area > 0.0005:
            # for triangle
            if len(approx) == 3:
                cnts_output.append(cnt)
                cv2.drawContours(out_image, [cnt], -1, (50, 255, 50), 7)
                cv2.fillPoly(out_image, pts =[cnt], color=(50, 255, 50))# ??
            # for quadrilater
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ratio = w / float(h)
                if(ratio >= 0.3 and ratio <= 3):
                    cnts_output.append(cnt)
                    cv2.drawContours(out_image, [cnt], -1, (50, 255, 50), 7)
                    cv2.fillPoly(out_image, pts =[cnt], color=(50, 255, 50))              
    return out_image

def blur(img,strength_blur = 7,sensitivity_to_light=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # binarize img
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if sensitivity_to_light != 'ignore':
        gray[gray>sensitivity_to_light] = 0
    blurred = cv2.medianBlur(gray,strength_blur)
    image_blurred = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
    return image_blurred

def get_contours(image):
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def resize(img, side,percent=50):
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
    if side:
        if side == 'right':
            img = img[:-50,470:-130]
        elif side == 'left':
            img = img[:-70,50:470]
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