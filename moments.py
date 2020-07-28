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

def resize(img, percent=20):
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

def find_moments(cnts, filename=None, hu_moment = True):
    '''
    this function returns the shape's Hu Moments.
    '''
    lst_moments = [cv2.moments(c) for c in cnts] # retrieve moments of all shapes identified
    lst_areas = [i["m00"] for i in lst_moments] # retrieve areas of all shapes
    
    max_idx = lst_areas.index(max(lst_areas)) # select shape with the largest area

    if hu_moment: # if we want the Hu moments instead
        HuMo = cv2.HuMoments(lst_moments[max_idx]) # grab humoments for largest shape
        if filename:
            HuMo = np.append(HuMo, filename)
        return HuMo

    # if we want to get the moments
    Moms = lst_moments[max_idx] 
    if filename:
        Moms['target'] = filename
    return Moms

def display_contour(cnts, img):
    for c in cnts:
        #compute the center of the contour
        M = cv2.moments(c)
        huM = cv2.HuMoments(M)
        # draw the contour and center of the shape on the image
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    # show the image
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def dist_humoment1(hu1,hu2):
    distance =  np.sum(abs(1/hu1-1/hu2))
    return distance

def dist_humoment2(hu1,hu2):
    distance =  np.sum(abs(hu1-hu2))
    # print(abs(hu1-hu2))
    return distance

def dist_humoment3(hu1,hu2):
    distance =  np.sum(abs(hu1-hu2)/abs(hu2))
    return distance

def save_moments(images):
    """
    compute moments for all images in our dataset
    """
    hu_moments = []
    moments = []

    for image_name, image_path in images.items():
        img_cv = cv2.imread(image_path)

        cnts, img = preprocess_img_test(img_cv)
        display_contour(cnts, img)
        hu_moments.append(find_moments(cnts, image_name))
        moments.append(find_moments(cnts, image_name, hu_moment=False))

        hu_moments_df = pd.DataFrame(hu_moments)
        hu_moments_df.to_csv('data/hu_moments.csv', index=False)

        moments_df = pd.DataFrame(moments)
        moments_df.to_csv('data/moments.csv', index=False)

    return hu_moments, moments
    
def test():
    hu_moments = pd.read_csv('data/hu_moments.csv')
    target = hu_moments.iloc[:, -1]

    # get 1 img
    images = get_files()

    # preprocess
    img_cv = cv2.imread(images['bateau'])
    cnts, img_cv = preprocess_img_test(img_cv)

    HuMo = find_moments(cnts)
    print(HuMo)

    proba = hu_moments.apply(lambda row : dist_humoment2(np.hstack(HuMo), row.values[:-1]), axis=1)
    proba_labelled = pd.concat([proba, target], axis=1)
    proba_labelled.columns = ['distance', 'target']
    print(proba_labelled.sort_values(by=["distance"]))

def read_video():
    """
    compare moments of each frame with the hu moments of our dataset images
    """

    hu_moments = pd.read_csv('data/hu_moments.csv')
    target = hu_moments.iloc[:, -1]

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()

        # Our operations on the frame come here
        cnts, img = preprocess_img(image, left_side=False)
        HuMo = np.hstack(find_moments(cnts))

        # get distances
        dist = hu_moments.apply(lambda row : dist_humoment2(HuMo, row.values[:-1]), axis=1)
        dist_labelled = pd.concat([dist, target], axis=1)
        dist_labelled.columns = ['distance', 'target']
        print(dist_labelled.sort_values(by=["distance"]))

        # Display the resulting frame
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test distance
    test()
