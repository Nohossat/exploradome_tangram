from .tangram_game import tangram_game
from .utils import get_files
import re
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# test statiques
def get_classification_report_pics(dataset_path=None, get_pred=tangram_game):
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
        predictions = get_pred(image = img_path)

        if predictions is None:
            continue

        if label != predictions.loc[0, 'classe']:
            print(img_path, label, predictions.loc[0, 'classe'])

        y_true.append(label)
        y_pred.append(predictions.loc[0, 'classe'])

    # get metrics
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, target_names=classes)
    print(type(report))

    # plot confusion matrix
    sns.heatmap(conf_matrix, annot = True, xticklabels=classes, yticklabels=classes)
    plt.show()

    return report


    

