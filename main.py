from tangram_app.tangram_game import tangram_game, tangram_game_live_test, tangram_game_dist
from tangram_app.metrics import get_classification_report_pics
from tangram_app.prepare_tangrams_dataset import get_files, save_moments
from tangram_app.prepro2 import *
from tangram_app.processing import *
import argparse
import os


if __name__ == '__main__':

    # paths available for testing
    path_vid = "data/videos/coeur.mov"
    path_img = "data/test_images/bateau_4_right.jpg"

    # cli options
    parser = argparse.ArgumentParser(description="Tangram detection\n")
    parser.add_argument('-m', '--mode', help='analyze picture or video', default=False) # default to webcam
    parser.add_argument('-s', '--side', help='analyze left / right or the full frame', default="left")
    parser.add_argument('-metrics', '--metrics', help='get metrics of our model', default=False)
    parser.add_argument('-test', '--test', help='test prepro', default=False)
    args = parser.parse_args()

    # check args.side value
    if args.side:
        assert args.side in ["right", "left", "none"], "Select a valid side : left - right"

    if args.mode:
        # check if mode is image / video or webcam : 0 : check how to connect to an ecternal webcam
        if args.mode.endswith((".jpg", ".png")):
            # it is an image => static testing
            assert os.path.exists(args.mode), "the file doesn't exist - try with another file"
            print(tangram_game(image=args.mode, side=args.side))
        elif args.mode.endswith((".mp4", ".mov")):
            # it is a video => live testing
            assert os.path.exists(args.mode), "the file doesn't exist - try with another file"
            tangram_game_live_test(video=args.mode, side=args.side)
        elif int(args.mode) == 0 or int(args.mode) == 1: # webcam
            tangram_game_live_test(video=int(args.mode), side=args.side, prepro=preprocess_img_2)
        else :
            raise Exception("the mode isn't valid - pass a valid image/video path or a webcam stream")
    
    if args.metrics :
        assert os.path.exists(args.metrics), "the folder doesn't exist - try with another one"
        report = get_classification_report_pics(dataset_path=args.metrics)
        print(report)

    if args.test:
        path = "data/test_images/cygne_20_left.jpg"
        path1 = "data/test_images/bateau_1_right.jpg"
        img_cv = cv2.imread(path1)

        print(tangram_game_dist(side="right", image=path1, prepro=False))
        # preprocess_img_original(img_cv, side="right")
        # print(path)
        # print(tangram_game(image=path, side="right", prepro=False))
    
    