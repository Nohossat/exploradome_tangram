import cv2
import numpy as np
import imutils
import pandas as pd
import os
from .distances import *
from .processing import *

def find_moments(cnts, filename=None, hu_moment = True):
    '''
    this function returns the shape's Moments or Hu Moments.

    =========

    Parameters : 

    cnts : contours of an image
    filename : name of the image, useful when target name needed
    hu_moment : returns Hu Moments. if set to False, returns only the Moments

    ========

    Returns : Moments or Hu moments

    ========
    author : @Bastien
    
    '''

    lst_moments = [cv2.moments(c) for c in cnts] # retrieve moments of all shapes identified
    lst_areas = [i["m00"] for i in lst_moments] # retrieve areas of all shapes
    
    try : 
        max_idx = lst_areas.index(max(lst_areas)) # select shape with the largest area

        if hu_moment: # if we want the Hu moments
            HuMo = cv2.HuMoments(lst_moments[max_idx]) # grab humoments for largest shape
            if filename:
                HuMo = np.append(HuMo, filename)
            return HuMo

        # if we want to get the moments
        Moms = lst_moments[max_idx] 
        if filename:
            Moms['target'] = filename
        return Moms
    except Exception as e:
        return [] # predictions impossible

def get_predictions(image, hu_moments, target, side=None, prepro=False):
    """
    compare moments of a frame with the hu moments of our dataset images  

    =========

    Parameters : 

    image : OpenCV image
    hu_moments : dataset with the humoments of each class
    target : name of the classes
    side : which side should be analyzed - left / right / full image

    ========

    Return : print the probabilities to belong to each class in descending order (Pandas DataFrame)

    author : @Nohossat
    """

    # Our operations on the frame come here
<<<<<<< HEAD:moments.py
    cnts, img = preprocess_img(image, side=side, crop = crop)
    HuMo = np.hstack(find_moments(cnts))
=======
    if not prepro : #take default prepro
        cnts = preprocess_img(image, side=side)
    else : # special prepro
        cnts = prepro(image, side=side)

    HuMo = find_moments(cnts)

    if len(HuMo) == 0 : 
        return None, None # the image can't be processed, so empty predictions

    # with the hu_moments we can get the predictions
    HuMo = np.hstack(HuMo)
>>>>>>> 7819ef5d13c73358fc33f06bd9cf88e1ecbe2752:tangram_app/moments.py

    # get distances
    dist = hu_moments.apply(lambda row : dist_humoment4(HuMo, row.values[:-1]), axis=1)
    dist_labelled = pd.concat([dist, target], axis=1)
    dist_labelled.columns = ['distance', 'target']

<<<<<<< HEAD:moments.py
    dist_labelled['proba'] =  round((1/dist_labelled['distance']) / np.sum( 1/dist_labelled['distance'], axis=0),2)
    #dist_labelled['proba'] =  round(np.exp(1/dist_labelled['distance']) / np.sum(np.exp(1/dist_labelled['distance']),axis=0),3)
    print(dist_labelled.sort_values(by=["proba"], ascending=False)[['target','proba']])
=======
    # get probabilities
    dist_labelled['proba'] = round((1/dist_labelled['distance']) / np.sum( 1/dist_labelled['distance'], axis=0),2)
    probas = dist_labelled.sort_values(by=["proba"], ascending=False)[['target','proba']]
    
    # sorted probabilities
    return probas.reset_index(drop=True), cnts
>>>>>>> 7819ef5d13c73358fc33f06bd9cf88e1ecbe2752:tangram_app/moments.py
