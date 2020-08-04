import cv2
import numpy as np
import imutils
import pandas as pd
import os
from .moments import get_predictions
from .prepare_tangrams_dataset import get_files


"""
main entry in the application: tangram_game
you can do live testing with the tangram_game_live_test
"""

def tangram_game(hu_moments_dataset='data/hu_moments.csv', crop=True, side=None, video=0, image=False):
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
    hu_moments = pd.read_csv(hu_moments_dataset)
    target = hu_moments.iloc[:, -1]

    # compare image with dataset images
    if image :
        assert os.path.exists(image), "the image doesn't exist"
        img_cv = cv2.imread(image)
        predictions = get_predictions(img_cv, hu_moments, target, side = side, crop = crop)
        return predictions

    # compare video frames with dataset images
    if not isinstance(video, bool):
        cap = cv2.VideoCapture(video)

        while(cap.isOpened()):
            ret, image = cap.read() # Capture frame-by-frame
            predictions = get_predictions(image, hu_moments, target, side = side, crop = crop)
            print(predictions)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

def tangram_game_live_test(crop=True, side=None, video=0):
    """
    analyze video stream to give the probabilities of the image / frame 
    to belong to each class of our dataset and display it

    =========

    Parameters : 

    video : gives the channel to watch. False by default
    side : the side to analyze on the frame : left / right / full frame (None)
    crop : crop image if raw image

    Returns : print predictions for each frame

    ========
    author : @Nohossat
    """

    # latest predictions - we record only the 100 latest predictions
    latest_predictions = []

    # get dataset
    hu_moments = pd.read_csv('data/hu_moments.csv')
    target = hu_moments.iloc[:, -1]

    # compare video frames with dataset images
    cap = cv2.VideoCapture(video)

    while(cap.isOpened()):
        ret, image = cap.read() # Capture frame-by-frame
        predictions = get_predictions(image, hu_moments, target, side = side, crop = crop)

        image = imutils.resize(image, width=1300)
        
        # add prediction on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        msg = "Classe predite : " + predictions.loc[0, 'target'] + " | Proba : " +str(predictions.loc[0, 'proba'])
        
        cv2.putText(image, msg, (350, 550), 
                    font, 0.8,
                    (89, 22, 76),  
                    2,
                    cv2.LINE_4)

        cv2.imshow('video', image)

        # add latest predictions
        latest_predictions.append(predictions.loc[0, 'target'])

        if len(latest_predictions) > 190:
            latest_predictions.remove(latest_predictions[0]) # get only the 390 latest predictions

        if latest_predictions.count(latest_predictions[0]) == 190: # if the latest predictions are equal, we check if the machine found the right label
            break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return latest_predictions[0]

if __name__ == '__main__':
    path = "data/tangrams/cygne.jpg"
    path2 = "/Users/nohossat/Documents/exploradome_videos/TangrIAm dataset/bol/bol.11.jpg"
    # print(tangram_game(image = path2, crop=False))

    # multi-processing here
    print(tangram_game(video=0, side="left"))
    print(tangram_game(video=0, side="right"))
