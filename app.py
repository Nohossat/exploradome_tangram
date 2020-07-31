import cv2
import numpy as np
import imutils
import pandas as pd
import os
from moments import get_predictions
from prepare_tangrams_dataset import get_files



# main application
# new comment by Nohossat

def tangram_game(crop, side=None, video=0, image=False):
    """
    analyze image or video stream to give the probabilities of the image / frame 
    to belong to each class of our dataset

    =========

    Parameters : 

    video : gives the channel to watch. False by default
    image : gives the filename of the image we want to predict. False by default
    side : the side to analyze on the frame : left / right / full frame (None)
    crop : crop image if raw image

    Returns : print predictions for each frame

    ========
    author : @Nohossat
    """

    # get dataset
    hu_moments = pd.read_csv('data/hu_moments.csv')
    target = hu_moments.iloc[:, -1]

    # compare image with dataset images
    if image :
        assert os.path.exists(image), "the image doesn't exist"
        img_cv = cv2.imread(image)
        predictions = get_predictions(img_cv, hu_moments, target, side = side, crop = crop)
        return predictions

    # compare video frames with dataset images
    if not isinstance(video, bool):
        cap = cv2.VideoCapture(video) # here it needs testing

        while(cap.isOpened()):
            ret, image = cap.read() # Capture frame-by-frame
            predictions = get_predictions(image, hu_moments, target, side = side, crop = crop)
            print(predictions)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print(tangram_game(image = 'data/tangrams/cygne.jpg', crop=False))
