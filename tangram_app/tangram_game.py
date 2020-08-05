import cv2
import numpy as np
import imutils
import pandas as pd
import os
from .moments import get_predictions
from .prepare_tangrams_dataset import get_files
<<<<<<< HEAD:tangram_app/tangram_game.py
import re
=======
>>>>>>> 37f95fb54c97b8a72a4615be2a7f817ee13b6a96:app.py


"""
main entry in the application: tangram_game
you can do live testing with the tangram_game_live_test
"""

<<<<<<< HEAD:tangram_app/tangram_game.py
def tangram_game(hu_moments_dataset='data/hu_moments.csv', side=None, video=0, image=False, prepro=False):
=======
def tangram_game(hu_moments_dataset='data/hu_moments.csv', crop=True, side=None, video=0, image=False):
>>>>>>> 37f95fb54c97b8a72a4615be2a7f817ee13b6a96:app.py
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

        # get size to analyze from image path
        pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
        result = pattern.search(image)
        side = result.group(2)

        # pass side and image to get predictions function
        img_cv = cv2.imread(image)
        predictions = get_predictions(img_cv, hu_moments, target, side = side, prepro=False) # retourne cnts so you can print them too
        return predictions

    # compare video frames with dataset images
    if not isinstance(video, bool):
        cap = cv2.VideoCapture(video)

        while(cap.isOpened()):
            ret, image = cap.read() # Capture frame-by-frame
            predictions = get_predictions(image, hu_moments, target, side = side) # retourne cnts so you can print them too
            print(predictions)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

def tangram_game_live_test(side=None, video=0):
    """
    analyze video stream to give the probabilities of the image / frame 
    to belong to each class of our dataset and display it

    =========

    Parameters : 

    video : gives the channel to watch. Webcam (0) by default
    side : the side to analyze on the frame : left / right / full frame (None)

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

    assert cap.isOpened(), "Unexpected error while reading video stream"

    while(cap.isOpened()):
        ret, image = cap.read()

        if not ret:
            print("we are done")
            break

        predictions = get_predictions(image, hu_moments, target, side = side)
        image = imutils.resize(image, width=1200)
        
        # add prediction on the frame
        if predictions is None:
            prediction_label = "N/A"
            prediction_proba = 0
            msg2 = ""
            msg3 = ""
        else:
            prediction_label = predictions.loc[0, 'target']
            prediction_proba = predictions.loc[0, 'proba']
            latest_predictions.append(prediction_label)

        msg = f"Main proba : {prediction_label}"

        if predictions is not None:
            msg += f" - {int(prediction_proba * 100)} %"
            msg2 = f"2 : {predictions.loc[1, 'target']} - {int(predictions.loc[1, 'proba'] * 100)} %"
            msg3 = f"3 : {predictions.loc[2, 'target']} - {int(predictions.loc[2, 'proba'] * 100)} %"
        
        # print messages
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(image, msg, (350, 450), 
                    font, 0.8,
                    (89, 22, 76),  
                    2,
                    cv2.LINE_4)

        if msg2 and msg3:
            cv2.putText(image, msg2, (350, 500), 
                        font, 0.6,
                        (89, 22, 76),  
                        1,
                        cv2.LINE_4)

            cv2.putText(image, msg3, (350, 525), 
                        font, 0.6,
                        (89, 22, 76),  
                        1,
                        cv2.LINE_4)

        # display frame
        cv2.imshow('tangram', image)
        cv2.moveWindow('tangram', 30, 30)

        # fine-tune the latest predictions in order to run the live testing code on several videos
        if len(latest_predictions) > 10:
            latest_predictions.remove(latest_predictions[0]) # get only the 190 latest predictions

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return latest_predictions[0]
