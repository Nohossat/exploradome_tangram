import re
import os

import seaborn as sns
import matplotlib.pyplot as plt
from .tangram_game import tangram_game
from .utils import get_files
from .processing import *
from .predictions import *
from sklearn.metrics import classification_report, confusion_matrix


def get_classification_report_pics(title_report="Tangram Game", dataset_path=None, game=tangram_game, prepro=preprocess_img_2, pred_func=get_predictions_with_distances):
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
        predictions = game(image=img_path, prepro=prepro, pred_func=pred_func)
        if predictions is None:
            continue

        if label != predictions.loc[0, 'target']:
            print(img_path, label, predictions.loc[0, 'target'])

        y_true.append(label)
        y_pred.append(predictions.loc[0, 'target'])

    # get metrics
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, target_names=classes)

    # plot confusion matrix
    conf_matrix_heatmap = sns.heatmap(conf_matrix, annot = True, xticklabels=classes, yticklabels=classes)

    # save report / matrix
    with open(f'metrics/{title_report}_report.txt', 'w') as f:
        f.write(report)

    fig = conf_matrix_heatmap.get_figure()
    fig.savefig(f'metrics/{title_report}_confusion_matrix.png')

    return report


    

