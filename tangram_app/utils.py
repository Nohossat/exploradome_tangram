from .processing import preprocess_img, display_contour
from .moments import find_moments
import pandas as pd
import os
import re
import cv2
import numpy as np

DATA_PATH = 'data/'

def get_files(directory = DATA_PATH + '/tangrams'):
    """
    get a dict with the image_name and the path of all images in tangrams folder
    author : @Nohossat
    """
    images = []
    assert os.path.exists(directory), "the directory doesn't exist"

    for folder, sub_folders, files in os.walk(directory):
        for file in files:
            filename, file_extension = os.path.splitext(file) # we just want the filename to save the path
            file_path = os.path.join(folder, file)
            if file.endswith((".jpg", ".png")) and not file.startswith('frame'):
                pattern = re.compile(r"[a-zA-Z]+") # in case there is any number or underscore in the name
                label = pattern.match(filename).group()
                images.append((label, file_path))

    return images
