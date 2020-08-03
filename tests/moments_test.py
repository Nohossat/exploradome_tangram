
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
 
from ..moments import *

#test returns the shape's Hu Moments
def test_find_humoments():
     img = '../data/tangrams/bateau.jpg'
     img_cv = cv2.imread(img)
     cnts, img = preprocess_img(img_cv, crop=False)
     humoments = find_moments(cnts, filename = 'bateau', hu_moment = True)
     assert humoments.shape == (8,) , 'Humoment is not correct'
    

def test_find_moments():
     img = '../data/tangrams/bateau.jpg'
     img_cv = cv2.imread(img)
     cnts, img = preprocess_img(img_cv, crop=False)
     moments = find_moments(cnts, filename = 'bateau', hu_moment = False)
     assert len(moments) == 25 , 'Moment is not correct'


    

    