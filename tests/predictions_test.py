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
 
from tangram_app.predictions import *

#def test_img_to_sorted_dists():
 #   pass

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

#def test_get_predictions_with_distances():
 #   pass