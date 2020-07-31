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
            cv2.waitKey(1)
            probas,area = get_predictions(image, hu_moments, target, side = side, crop = crop)
            
            # Below this, test
            mapper = {0:"cygne",1:'bateau',2:'renard',3:'maison',4:'marteau',5:'tortue',6:'pont',7:'lapin',8:'coeur',9:'chat',10:'montagne',11:'bol'}

            font = cv2.FONT_HERSHEY_SIMPLEX
            if area < .3:
                msg = "Classe predite : " + mapper[probas.sort_values(ascending=False).index[0]] + " | Proba : " +str(probas.sort_values(ascending=False).values[0])
            else:
                msg = "Plateau trop charge, impossible de predire"

            cv2.putText(image,
                msg,
                (500, 500),
                font, 1,
                (0, 255, 255),  
                2,
                cv2.LINE_4)
            cv2.imshow('video', image)

            print(probas.sort_values(ascending=False),area)
            # print(get_predictions(image, hu_moments, target, side = side, crop = crop))

            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
