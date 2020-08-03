"""
@author: Renata
"""

import pytest
from ..find_corner import *

#test visualize on an image the corner point in red
def test_get_nb_corners():
    img = '../data/tangrams/bateau.jpg'
    img_cv = cv2.imread(img)
    assert get_nb_corners(img_cv) == 24 , 'Nombre of corners is incorrect'