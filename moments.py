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
        ret, frame = cap.read()

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
