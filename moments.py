import cv2
import numpy as np
import imutils
import pandas as pd
import os

images = ['bateau', 'chat', 'coeur', 'cygne', 'lapin', 'maison', 'marteau', 'montagne', 'pont', 'renard', 'bol']

def img_preprocessing(image, path = None):
    # load the image, convert it to grayscale, blur it slightly,    
    # and threshold it : à quoi ça sert le threshold ?
    if path:
        image = cv2.imread(f'tangrams/{image}.JPG')
    else :
        image = cv2.imread(image)
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts

def get_moment(image, path):
    """
    from an Open CV image, get its moments and return them
    """
    moments = []
    cnts = img_preprocessing(image, path = path)

    for c in cnts:
        M = cv2.moments(c)
        moments.append(M)
    return moments

def save_moments():
    for image in images:
        moments = get_moment(image, path = True)
    return moments

def compare_moments_with_labels():
    """
    get CSV file and compare the moments from each class with the moments of the frame
    """
    pass

def read_video():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()

        # Our operations on the frame come here
        cnts = img_preprocessing(image)
        moments = get_moment(image)

        proba = compare_moments_with_labels()

        print(proba)

        # Display the resulting frame
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # if we call this file directly, we compute the moments for all the classes
    # save moments for each class
    moments_df = pd.DataFrame(save_moments())
    moments_df.to_csv('moments.csv', index=False)











def resize_img(img,left_side=True):
    img =resize(img,20)[0:-50, 55:-100].copy() # reisze img to 20% and crop to keep only board
    if left_side == True:
        img = img[0:int(img.shape[0]),0:int(img.shape[1]/2)] # keep only the left half of the board
    elif left_side == False:
        img = img[0:int(img.shape[0]),int(img.shape[1]/2):] # keep only the right half of the board
    return img

def find_humos(img,sensitivity_to_light=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #binarize img
    gray[gray>sensitivity_to_light] = 0 # turn background to black
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    lst_moments = [cv2.moments(c) for c in cnts] # retrieve moments of all shapes identified
    lst_areas = [i["m00"] for i in lst_moments] # retrieve areas of all shapes
    
    max_idx = lst_areas.index(max(lst_areas)) # select shape with the largest area
    HuMo = cv2.HuMoments(lst_moments[max_idx]) # grab humoments for largest shape
    return HuMo