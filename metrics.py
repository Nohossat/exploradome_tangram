from app import tangram_game, tangram_game_live_test
from prepare_tangrams_dataset import get_files
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

# test statiques
def get_classification_report(dataset_path=None):
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

    for label, img_path in images: 
        predictions = tangram_game(image = img_path, crop=False)

        if predictions is None:
            continue

        y_true.append(label)
        y_pred.append(predictions.loc[0, 'target'])

    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, target_names=classes)

    return report


# test dynamiques
def get_classification_report_videos(target_name, video_path=0):
    
    correct_predictions = 0
    prediction = tangram_game_live_test(crop=True, side="left", video=video_path)

    if prediction == target_name:
        print(prediction, target_name)
        correct_predictions += 1
    
    return f'label : {target_name} \n correct_predictions: {correct_predictions}'


if __name__ == "__main__":
    path = "/Users/nohossat/Documents/exploradome_videos/TangrIAm dataset"
    path2 = "/Users/nohossat/Documents/exploradome_videos/photos"
    path3 = "/Users/nohossat/Documents/exploradome_videos/TangrIAm dataset/yolo-team1/yolo-team1/train"
    # print(get_classification_report(dataset_path=path))

    video = "/Users/nohossat/Documents/exploradome_videos/videos/bol.mov"
    video1 = "/Users/nohossat/Documents/exploradome_videos/video_exploradome.mp4"
    video2 = "/Users/nohossat/Documents/exploradome_videos/videos/tortue_1.mov"
    print(get_classification_report_videos(target_name="bol", video_path=video))

