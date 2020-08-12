import numpy as np
import os
import cv2
import imutils


def preprocess_img(img, side=None, sensitivity_to_light=100):
    '''
    this function takes a cv image as input, calls the resize function, crops the image to keep only the board, chooses the left / right half of the board or the full board if the child is playing alone, and eventually finds the largest dark shape
    
    Parameters : 
    img = OpenCV image
    side = process either left/right side or full frame.  - True by default
    crop = decides if image needs cropping - set crop to False when processing dataset images, they are already cut
    sensitivity_to_light = parameter to turn the background black
    
    author : @BasCR-hub
    '''

    img = crop(img, side=side)
    image_blurred = blur(img,1)
    cnts = get_contours(image_blurred)
    image_triangles_squares = extract_triangles_squares(cnts, img)
    blurred_triangles_squared = blur(image_triangles_squares, 7, sensitivity_to_light='ignore').copy()
    final_cnts = get_contours(blurred_triangles_squared)

    return final_cnts, img

def preprocess_img_2(origin_img, side):
    '''
    this function takes a cv image as input, crops it to focus on the right/left side, detects edges using OpenCV's Canny, 
    dilates the resulting contours to make them continuous, binarizes the image and finds contours.
    It then calls the extract_triangles_squares_function to keep only regular geometric shapes (and eliminate all "noise" contours)
    
    Parameters : 
    origin_img = OpenCV image
    side = process either left/right side or full frame. Passed from the CLI argument
    
    author : @BasCR-hub
    '''

    origin_img = crop(origin_img, side=side)
    img = cv2.Canny(origin_img, 30, 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    img = cv2.dilate(img, kernel)
    img = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_BINARY)[1]
    cnts, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts_output, triangle_squares_img = extract_triangles_squares_2(cnts, img)
    return cnts_output, origin_img

def extract_triangles_squares(cnts, image):  
    '''
    This function extracts only the triangles, squares and parallelograms from a list of contours and hence removes 'noise' items.
    To do so, it calls OpenCV's appoxPolyDp function to identify which contours have either 3 or 4 summits then returns those that 
    fit area and/or width-height ratio criteria.
    
    Parameters :
    cnts_output = list of contours from cv2.findContours
    origin_img = OpenCV image
    
    author : @Gautier
    '''  
    cnts_output = []
    out_image = np.zeros(image.shape, image.dtype)

    for idx, cnt in enumerate(cnts):
        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetre, True)

        area = cv2.contourArea(cnt)
        img_area = image.shape[0] * image.shape[1]

        if area/img_area > 0.0005:
            # for triangle
            if len(approx) == 3:
                cnts_output.append(cnt)
                cv2.drawContours(out_image, [cnt], -1, (50, 255, 50), 7)
                cv2.fillPoly(out_image, pts =[cnt], color=(50, 255, 50))
            # for quadrilater
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ratio = w / float(h)
                if(ratio >= 0.3 and ratio <= 3):
                    cnts_output.append(cnt)
                    cv2.drawContours(out_image, [cnt], -1, (50, 255, 50), 7)
                    cv2.fillPoly(out_image, pts =[cnt], color=(50, 255, 50))         
    return out_image

def blur(img, strength_blur = 7, sensitivity_to_light=50):
    '''
    This function takes a cv image as input, turns it into grayscale and blurs it. Used before finding contours to erase the white space between individual shapes.
    Parameters : 
    
    origin_img = OpenCV image
    strength_blur = blur paramet (the greater the more "blurred")
    
    author : @BasCR-hub
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # binarize img
    if sensitivity_to_light != 'ignore':
        gray[gray > sensitivity_to_light] = 0
    blurred = cv2.medianBlur(gray, strength_blur)
    image_blurred = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
    return image_blurred

def get_contours(image):
    '''
    This function retrieves an image's openCV contours.
    
    Parameters : 
    image = OpenCV image
    
    author : @BasCR-hub
    '''

    cnts = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def display_contour(cnts, img):
    """
    display the contour of the image
    

    Parameters : 
    cnts : contours of the forms in the image
    img : OpenCV image
    
    author : @BasCR-hub
    """
    for c in cnts:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

    img = imutils.resize(img, width=1200)
    cv2.imshow("Image", img)
    cv2.moveWindow('Image', 30, 30)
    cv2.waitKey(0)

def extract_triangles_squares_2(cnts, img):
    '''
    This function extracts only the triangles, squares and parallelograms from a list of contours and hence removes 'noise' items.
    To do so, it calls OpenCV's appoxPolyDp function to identify which contours have either 3 or 4 summits then returns those that 
    fit area and/or width-height ratio criteria.
    
    Parameters :
    cnts_output = list of contours from cv2.findContours
    origin_img = OpenCV image
    
    author : @Gautier
    '''
    
    cnts_output = []
    out_image = np.zeros(img.shape, img.dtype)
    
    for idx,cnt in enumerate(cnts):
        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetre, True)
        area = cv2.contourArea(cnt)
        img_area = img.shape[0] * img.shape[1]

        if area/img_area > 0.0005:
            # for triangle
            if len(approx) == 3:
                cnts_output.append(cnt)
                cv2.drawContours(img, [cnt], -1, (50, 255, 50), 3)
                cv2.fillPoly(img, pts=[cnt], color=(50, 255, 50))  # ??
            # for quadrilater
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ratio = w / float(h)
                if(ratio >= 0.2 and ratio <= 4):
                    cnts_output.append(cnt)
                    cv2.drawContours(img, [cnt], -1, (50, 255, 50), 3)
                    cv2.fillPoly(img, pts=[cnt], color=(50, 255, 50))
    
    return cnts_output, out_image

def crop(img, side="left"):
    """
    crop the left or right side of the image 

    Parameters:
    - img : OpenCV
    - side : left / right

    Returns : OpenCV image

    author: @Nohossat
    """

    if side is None:
        return img

    assert side in ["left", "right"], "not a valid side"

    # we take only 55% of the frame either left or right side
    width_img = img.shape[1]
    box_width = int(width_img*0.55)

    if side == 'left':
        img = img[:, :box_width]
    else:
        box_width = width_img - box_width
        img = img[:, box_width:width_img]

    return img
