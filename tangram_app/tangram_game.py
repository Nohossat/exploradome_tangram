import cv2
import numpy as np
import imutils
import pandas as pd
import os
import re
import pprint

# files from the app
from .processing import preprocess_img
from .predictions import *


"""
main entry in the application: tangram_game
you can do live testing with the tangram_game_live_test
"""

def tangram_game(side=None, video=0, image=False, prepro=preprocess_img, pred_func=get_predictions):
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

    # compare image with dataset images
    if image :
        assert os.path.exists(image), "the image doesn't exist"

        # get size to analyze from image path
        pattern = re.compile(r"([a-zA-Z]+)_\d{1,2}_(\w+)")
        result = pattern.search(image)
        side = result.group(2)

        # pass side and image to get predictions function
        img_cv = cv2.imread(image)
        predictions = pred_func(img_cv, side = side, prepro=prepro)
        
        # display image
        # display_img(img_cv)

        return predictions

    # compare video frames with dataset images
    if not isinstance(video, bool):
        cap = cv2.VideoCapture(video)

        assert cap.isOpened(), "Unexpected error while reading video stream"

        while(cap.isOpened()):
            ret, image = cap.read() # Capture frame-by-frame

            if not ret:
                print("we are done")
                break

            predictions = pred_func(image, side = side, prepro=prepro)
            display_predictions(predictions, image, onscreen=True)

            if cv2.waitKey(3) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

def display_predictions(predictions, image, onscreen=True):
    """
    if onscreen True, we display the main proba on screen, otherwise we just print the 12 sorted probabilities
    """
    if onscreen:
        image = imutils.resize(image, width=1200)
        
        if predictions is None:
            prediction_label = "N/A"
            prediction_proba = 0
            msg2 = ""
            msg3 = ""
        else:
            prediction_label = predictions.loc[0, 'target']
            prediction_proba = predictions.loc[0, 'proba']

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
    else :
        print(predictions)

def display_img(img):
    cv2.imshow("Predictions", img)
    cv2.moveWindow("Predictions", 30, 30)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
