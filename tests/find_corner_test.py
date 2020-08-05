"""
@author: Renata
"""

import pytest
from tangram_app.find_corner import *

#test visualize on an image the corner point in red
def test_get_nb_corners():
    img = 'data/tangrams/bateau_4_right.jpg'
    img_cv = cv2.imread(img)
    assert get_nb_corners(img_cv) == 74, 'Nombre of corners is incorrect'
