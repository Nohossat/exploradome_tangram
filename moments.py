import cv2
import numpy as np
import imutils
import pandas as pd
import os
from distances import *
from processing import *

def find_moments(cnts, filename=None, hu_moment = True):
    '''
    this function returns the shape's Hu Moments.
    author : @Bastien
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

def save_moments(images):
    """
    compute moments for all images in our dataset
    author : @Nohossat
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
    
def get_distance_img_test(image):
    """
    test the main algorithm on 1 image
    author : @Nohossat
    """
    hu_moments = pd.read_csv('data/hu_moments.csv')
    target = hu_moments.iloc[:, -1]

    # get 1 img
    images = get_files()

    # preprocessing the image
    img_cv = cv2.imread(images[image])
    cnts, img_cv = preprocess_img_test(img_cv)

    # compute distance
    HuMo = find_moments(cnts)
    proba = hu_moments.apply(lambda row : dist_humoment2(np.hstack(HuMo), row.values[:-1]), axis=1)
    proba_labelled = pd.concat([proba, target], axis=1)
    proba_labelled.columns = ['distance', 'target']
    print(proba_labelled.sort_values(by=["distance"]))

def read_video(video=False):
    """
    compare moments of each frame with the hu moments of our dataset images
    author : @Nohossat
    """

    hu_moments = pd.read_csv('data/hu_moments.csv')
    target = hu_moments.iloc[:, -1]

    if video:
        cap = cv2.VideoCapture(video)
    else : 
        cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, image = cap.read()

        if ret :
            print(ret)

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
    get_distance_img_test('cygne') # only for testing
