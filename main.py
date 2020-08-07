from tangram_app.tangram_game import tangram_game
from tangram_app.metrics import get_classification_report_pics
from tangram_app.predictions import img_to_sorted_dists, get_predictions
from tangram_app.processing import preprocess_img, preprocess_img_2
import argparse
import os
import cv2


if __name__ == '__main__':

    # paths available for testing
    path_vid = "data/videos/coeur.mov"
    path_img = "data/test_images/bateau_4_right.jpg"

    # cli options
    parser = argparse.ArgumentParser(description="Tangram detection\n")
    # default to webcam
    parser.add_argument(
        '-m', '--mode', help='analyze picture or video', default=False)
    parser.add_argument(
        '-s', '--side', help='analyze left / right or the full frame', default="left")
    parser.add_argument('-metrics', '--metrics',
                        help='get metrics of our model', default=False)
    parser.add_argument('-test', '--test', help='test prepro', default=False)
    args = parser.parse_args()

    # check args.side value
    if args.side:
        assert args.side in ["right", "left",
                             "none"], "Select a valid side : left - right"

    if args.mode:
        if args.mode.endswith((".jpg", ".png")):
            # static testing
            assert os.path.exists(args.mode), "the file doesn't exist - try with another file"
            print(tangram_game(image=args.mode, side=args.side, prepro=preprocess_img_2, pred_func=img_to_sorted_dists))
        elif args.mode.endswith((".mp4", ".mov")): 
            # live testing
            assert os.path.exists(args.mode), "the file doesn't exist - try with another file"
            tangram_game(video=args.mode, side=args.side, prepro=preprocess_img_2, pred_func=img_to_sorted_dists)
        elif int(args.mode) == 0 or int(args.mode) == 1: 
            # webcam
            tangram_game(video=int(args.mode), side=args.side, prepro=preprocess_img_2, pred_func=img_to_sorted_dists)
        else :
            raise Exception("the mode isn't valid - pass a valid image/video path or a webcam stream")
    
    if args.metrics :
        assert os.path.exists(args.metrics), "the folder doesn't exist - try with another one"
        report = get_classification_report_pics(dataset_path=args.metrics, get_pred=tangram_game)
        print(report)

    if args.test:
        path = "data/test_images/cygne_20_left.jpg"
        path1 = "data/test_images/bateau_1_right.jpg"
        img_cv = cv2.imread(path1)
        print(tangram_game(side="right", image=path, prepro=preprocess_img_2, pred_func=img_to_sorted_dists))
    
    
