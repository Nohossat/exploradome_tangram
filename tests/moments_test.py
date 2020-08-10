
"""
Created on Tue Jul 28 15:11:06 2020

@author: Renata
"""

import pytest
import cv2
import numpy as np
import imutils
import pandas as pd
import os
import re
 
from tangram_app.moments import *

#test returns the shape's Hu Moments
''''def test_find_humoments():
     img = 'data/tangrams/bateau_4_right.jpg'
     pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
     result = pattern.search(img)
     side = result.group(2)
     img_cv = cv2.imread(img)
     cnts= preprocess_img(img_cv, side=side)
     humoments = find_moments(cnts, filename = 'bateau', hu_moment = True)
     assert humoments.shape == (8,) , 'Humoment is not correct'
    '''

def test_find_moments():
     img = 'data/tangrams/bateau_4_right.jpg'
     pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
     result = pattern.search(img)
     side = result.group(2)
     img_cv = cv2.imread(img)
     cnts = preprocess_img(img_cv, side=side)
     moments = find_moments(cnts, filename = 'bateau', hu_moment = False)
     assert len(moments.keys()) == 25 , 'Moment is not correct'


    
def test_save_moments():
# test moments and hu_moments into CSV files and return them as Pandas dataframe
    images = get_files(directory='data/tangrams') # we have to change the images in this one
    humoments, moments = save_moments(images, directory = 'tests/data')
    assert isinstance(humoments, pd.core.frame.DataFrame), 'Humoments are not correct'
    assert isinstance(moments, pd.core.frame.DataFrame), 'Moments are not correct'
    # assert os.path.exists('data/hu_moments.csv')
    # assert os.path.exists('data/moments.csv')
