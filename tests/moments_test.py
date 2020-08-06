
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
def test_find_humoments():
     img = 'data/tangrams/bateau_4_right.jpg'
     pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
     result = pattern.search(img)
     side = result.group(2)
     img_cv = cv2.imread(img)
     cnts= preprocess_img(img_cv, side=side)
     humoments = find_moments(cnts, filename = 'bateau', hu_moment = True)
     assert humoments.shape == (8,) , 'Humoment is not correct'
    

def test_find_moments():
     img = 'data/tangrams/bateau_4_right.jpg'
     pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
     result = pattern.search(img)
     side = result.group(2)
     img_cv = cv2.imread(img)
     cnts = preprocess_img(img_cv, side=side)
     moments = find_moments(cnts, filename = 'bateau', hu_moment = False)
     assert len(moments.keys()) == 25 , 'Moment is not correct'


def test_get_predictions():
    # test the probabilities to belong to each class in descending order
     img = 'data/tangrams/bateau_4_right.jpg'
     img_cv = cv2.imread(img)
     humoments = pd.read_csv('data/hu_moments.csv')
     target = humoments.iloc[:, -1]
     probalitity, cnts = get_predictions(img_cv, humoments, target)
     assert isinstance(probalitity, pd.core.frame.DataFrame), 'Predictions should be dataframe'
     assert isinstance(cnts, list), 'Contours should be list'
     assert probalitity.loc[0, 'target'] == 'bateau', 'Predictions should be bateau'
    