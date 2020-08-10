import re
import os

import seaborn as sns
import matplotlib.pyplot as plt
from .tangram_game import tangram_game
from .utils import get_files
from .processing import *
from .predictions import *
from sklearn.metrics import classification_report, confusion_matrix

# test statiques
def get_classification_report_pics(dataset_path=None, game=tangram_game):
    """
    from a set of images, get global accuracy, precision, recall
    """
    total_predictions = 0
    y_true = []
    y_pred = []
    classes = ["bateau", "bol", "chat", "coeur", "cygne", "lapin", "maison", "marteau", "montagne", "pont", "renard", "tortue"]

    if dataset_path is None:
        images = get_files() # get images in data/tangrams
    else :
        images = get_files(directory=dataset_path)


    # for each image, get prediction by our algorithm
    for label, img_path in images: 
        predictions = game(image=img_path, prepro=preprocess_img_2, pred_func=get_predictions_with_distances)
        if predictions is None:
            continue

        if label != predictions.loc[0, 'target']:
            print(img_path, label, predictions.loc[0, 'target'])

        y_true.append(label)
        y_pred.append(predictions.loc[0, 'target'])

    # get metrics
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, target_names=classes)
    print(type(report))

    # plot confusion matrix
    sns.heatmap(conf_matrix, annot = True, xticklabels=classes, yticklabels=classes)
    plt.show()

    return report


    

