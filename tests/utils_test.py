"""
@author: Renata
"""

import pytest
import cv2
import numpy as np
import imutils
import pandas as pd
import os
import re

from tangram_app.utils import *

def test_get_files():
    assert len(get_files(directory='data/tangrams')) == 12, 'Images must be 12'
