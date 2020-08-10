import numpy as np
import os
import cv2
import imutils


def preprocess_img(img, side=None, sensitivity_to_light=50):
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
    image_blurred = blur(img,3)
    final_cnts = get_contours(image_blurred)
    # image_triangles_squares = extract_triangles_squares(cnts, img)
    # blurred_triangles_squared = blur(image_triangles_squares, 3, sensitivity_to_light='ignore').copy()
    # final_cnts = get_contours(blurred_triangles_squared)
    return final_cnts, img

def preprocess_img_2(origin_img, side):
    origin_img = crop(origin_img, side=side)
    img = cv2.Canny(origin_img, 30, 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    img = cv2.dilate(img, kernel)
    img = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_BINARY)[1]
    cnts, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts_output, triangle_squares_img = extract_triangles_squares_2(cnts, img)

    # for c in cnts:
        # cv2.drawContours(origin_img, [c], -1, (50, 255, 50), 1)

    # cv2.imshow("image", origin_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cnts_output, origin_img

def extract_triangles_squares(cnts, image):    
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # binarize img
    if sensitivity_to_light != 'ignore':
        gray[gray > sensitivity_to_light] = 0
    blurred = cv2.medianBlur(gray, strength_blur)
    image_blurred = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
    return image_blurred

def get_contours(image):
    cnts = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def resize(img, side, percent=50):
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
    scale_percent = percent  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA).copy()

    if side:
        if side == 'right':
            img = img[:-50, 470:-130]
        elif side == 'left':
            img = img[:-70, 50:470]
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

    img = imutils.resize(img, width=1200)
    cv2.imshow("Image", img)
    cv2.moveWindow('Image', 30, 30)
    cv2.waitKey(0)

def extract_triangles_squares_2(cnts, img):    
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

    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cnts_output, out_image

def crop(img, side="left"):
    """
    crop the left or right side of the image 

    Parameters:
    - img : OpenCV
    - side : left / right

    Returns : OpenCV image
    """
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

def contour_intersect(main_cnt, new_cnt):
    """
    check if contour new_cnt intersect with the main one (main_cnt)
    """
    intersecting_pts = []

    ## Loop through all points in the contour
    for pt in new_cnt:
        x,y = pt[0]

        ## find point that intersect the reference contour
        ## edges_only flag check if the intersection to detect is only at the edges of the contour
        i = 0
        for cnt in main_cnt:
            if [[x, y]] in cnt:
                if i == 0:
                    print([[x, y]])
                    i += 1
                intersecting_pts.append(pt[0])

    if len(intersecting_pts) > 0:
        return True
    else:
        return False

def detect_black_color(img):
    # img : OpenCV image
    BLACK_MIN = np.array([0, 0, 0])
    BLACK_MAX = np.array([180, 255, 29])

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    frame_threshed = cv2.inRange(hsv_img, BLACK_MIN, BLACK_MAX)

    return frame_threshed

def detect_white_color(img):
    # img : OpenCV image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,168], dtype=np.uint8)
    upper_white = np.array([172,111,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    return mask

