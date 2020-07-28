import cv2
import numpy as np
import imutils
import pandas as pd
import os
import time

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

def preprocess_img(img,left_side=True):
    '''
    this function takes a cv image as input, calls the resize function then crops the image to keep only the board.
    It then chooses the left or right half of the board, based on user input 
    '''

    img =resize(img,20)[0:-50, 55:-100].copy() # reisze img to 20% and crop to keep only board
    if left_side == True:
        img = img[0:int(img.shape[0]),0:int(img.shape[1]/2)] # keep only the left half of the board
    elif left_side == False:
        img = img[0:int(img.shape[0]),int(img.shape[1]/2):] # keep only the right half of the board
    return img

def find_humos(img,filename, sensitivity_to_light=50):
<<<<<<< HEAD
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # binarize img
=======
    '''
    this function finds the largest dark shape in the image and returns the shape's Hu Moments.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #binarize img
>>>>>>> ffa73666f2c9f77e04c4186e1e5f15871bf74f85
    gray[gray>sensitivity_to_light] = 0 # turn background to black
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    lst_moments = [cv2.moments(c) for c in cnts] # retrieve moments of all shapes identified
    lst_areas = [i["m00"] for i in lst_moments] # retrieve areas of all shapes
    
    max_idx = lst_areas.index(max(lst_areas)) # select shape with the largest area
    HuMo = cv2.HuMoments(lst_moments[max_idx]) # grab humoments for largest shape
    HuMo = np.append(HuMo, filename)
    return HuMo

def resize(img,percent=20):
    '''
    this function takes a cv image as input and resizes it. The primary objective is to make the contouring less sensitive to between-tangram demarcation lines,
    the secondary objective is to speed up processing.
    '''
    scale_percent = percent # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA).copy()
    return img

def save_moments(images):
    """
    compute moments for all images in our dataset
    """
    hu_moments = []
    for image_name, image_path in images.items():
        image = cv2.imread(image_path)
        hu_moments.append(find_humos(image, image_name))
    return np.array(hu_moments).reshape(12, 8)

def compare_moments_with_labels():
    """
    get CSV file and compare the moments from each class with the moments of the frame
    """
    pass

def read_video():
    """
    compare moments of each frame with the hu moments of our dataset images
    """
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
    images = get_files()
    hu_moments = save_moments(images)
    hu_moments_df = pd.DataFrame(hu_moments)
    hu_moments_df.to_csv('hu_moments.csv', index=False)
