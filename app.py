import cv2
import numpy as np
import imutils
import pandas as pd
from moments import get_predictions
from prepare_tangrams_dataset import get_files



# main application
# new comment by Nohossat

def tangram_game(side, crop, video=0, image = False):
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
        images = get_files()
        image = cv2.imread(images[image])
        print(get_predictions(image, hu_moments, target, side = side, crop = crop))

    # compare video frames with dataset images
    if not isinstance(video, bool):
        cap = cv2.VideoCapture(video) # here it needs testing

        while(cap.isOpened()):
            ret, image = cap.read() # Capture frame-by-frame
            print(get_predictions(image, hu_moments, target, side = side, crop = crop))

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tangram_game(video=False, image = 'cygne', side="left", crop=False)
