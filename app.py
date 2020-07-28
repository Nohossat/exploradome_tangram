import cv2
import numpy as np
import imutils
import pandas as pd
import moments as mm
import os
from distances import *
from processing import *

# main application

# read video and get frame by frame the probabilities to belong to each class
if __name__ == '__main__':
    mm.read_video() # do all the job for now
