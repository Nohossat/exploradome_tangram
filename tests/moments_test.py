
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
def test_find_moments():
     assert find_moments(cnts, filename=None, hu_moment = True) ==  np.append(HuMo, filename), 'Moment is not correct'
    

def test_get_predictions():
    # test the probabilities to belong to each class in descending order
   assert get_predictions(image, hu_moments, target, side=None, crop = True) == probas.reset_index(drop=True) , 'Predictions is not correct'
'''
    #verification de path of vidéo
    if __name__ == '__main__':
    #test distance
    get_distance_img_test('cygne)'''




    

    