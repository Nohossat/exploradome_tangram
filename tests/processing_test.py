
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
 
import sys
sys.path.insert(0, '/exploradome_tangram-numpy-team-4')

import processing.py

#test if get images is in jpg format
def get_files_test():
    dicto = {'bateau.jpg', 'bol.jpg', 'chat.jpg', 'coeur.jpg', 'cygne.jpg', 'lapin.jpg', 'maison.jpg', 'martaeu.jpg', 'montagne.jpg', 'pont.jpg', 'renard.jpg', 'tortue.jpg'
    }
    assert get_files() == dicto
    
#test a cv image and resize operations
def preprocess_img_test():
    from PIL import Image
try:
    im=Image.open(filename)
    # do stuff
except IOError:
    # filename not an image file
    # verification extension of file *.img

#test cv image and resizes it. 
def resize_test():
    os.path.exists(dirname), "the directory doesn't exist"

#test the contour of the image
def display_contour(cnts, img):
    os.path.exists(dirname), "the directory doesn't exist"
    
