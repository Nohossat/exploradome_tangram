import cv2
import numpy as np
import imutils
import pandas as pd
import os
import re

from tangram_app.processing import preprocess_img

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

def save_moments(images, directory):
    """
    compute moments / hu moments for all images in our dataset

    Parameters : 

    images : dict with images names and paths

    ========
    Return : save moments and hu_moments into CSV files and return them as Pandas dataframe

    author : @Nohossat
    """

    hu_moments = []
    moments = []

    for image_name, image_path in images:
        img_cv = cv2.imread(image_path)

        pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")

        result = pattern.search(image_path)

        if result :
            side = result.group(2)
        else :
            side = None

        cnts, img = preprocess_img(img_cv, side=side)

        hu_moments.append(find_moments(cnts, image_name))
        moments.append(find_moments(cnts, image_name, hu_moment=False))

        hu_moments_df = pd.DataFrame(hu_moments)
        hu_moments_df.to_csv(directory +'/hu_moments.csv', index=False)

        moments_df = pd.DataFrame(moments)
        moments_df.to_csv(directory + '/moments.csv', index=False)

    return hu_moments_df, moments_df
    
