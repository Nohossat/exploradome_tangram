import imutils


def preprocess_img(img, side=None, crop = True, sensitivity_to_light=50):
    img = resize(img).copy()
    if crop :
        if side == "left":
            img = img[0:int(img.shape[0]),int(img.shape[1]/2):] # keep only the left half of the board
        elif side == "right" :
            img = img[0:int(img.shape[0]),0:int(img.shape[1]/2)] # keep only the right half of the board
        else:
            img = img[0:-50, 55:-100] # get full frame, if child plays alone  
            
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # binarize img
    gray[gray>sensitivity_to_light] = 0 # turn background to black
    blurred = cv2.medianBlur(gray,3)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]  # ??
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) # we need the contours to compute Hu moments
    return cnts, img

def resize(img, percent=50):
    scale_percent = percent # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA).copy()
    img = img[:,:465]
    return img

def detect_forme(cnts, image):    
    cnts_output = []
    out_image = np.zeros(image.shape, image.dtype)
    for idx,cnt in enumerate(cnts):
        perimetre = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetre, True)

        area = cv2.contourArea(cnt)
        img_area = image.shape[0] * image.shape[1]
        
        if area/img_area > 0.0005:
            # for triangle
            if len(approx) == 3:
                cnts_output.append(cnt)
                cv2.drawContours(out_image, [cnt], -1, (50, 255, 50), 2)
                cv2.fillPoly(out_image, pts =[cnt], color=(50, 255, 50))# ??
            # for quadrilater
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ratio = w / float(h)
                if(ratio >= 0.3 and ratio <= 3):
                    cnts_output.append(cnt)
                    cv2.drawContours(out_image, [cnt], -1, (50, 255, 50), 2)
                    cv2.fillPoly(out_image, pts =[cnt], color=(50, 255, 50))
                    
    return out_image

def blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # binarize img
    blurred = cv2.medianBlur(gray,7)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
    return thresh

def get_contours(image):
    new_cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    new_cnts = imutils.grab_contours(new_cnts)
    for c in new_cnts:
        cv2.drawContours(image, [c], -1, (50, 255, 50), 2)
    return new_cnts

def get_humos(new_cnts):
    lst_moments = [cv2.moments(c) for c in new_cnts] # retrieve moments of all shapes identified
    lst_areas = [i["m00"] for i in lst_moments] # retrieve areas of all shapes
    max_idx = lst_areas.index(max(lst_areas)) # select shape with the largest area

    HuMo = cv2.HuMoments(lst_moments[max_idx]) # grab humoments for largest shape
    HuMo = np.hstack(HuMo)
    return HuMo

def get_distance(HuMo):
    hu_moments = pd.read_csv("hu_moments.csv")
    target = hu_moments.iloc[:,-1]
    dist = hu_moments.apply(lambda row : dist_humoment(HuMo, row.values[:-1]), axis=1)
    dist_labelled = pd.concat([dist, target], axis=1)
    dist_labelled.columns = ['distance', 'target']
    dist_labelled = dist_labelled.sort_values(by=['distance'],ascending=True)
    return dist_labelled

def dist_humoment(hu1,hu2):
    distance =  np.sum(abs(hu1-hu2))
    return distance


lst_images = listdir(r"C:\Users\bcarniel\Desktop\Simplon\exploradome\static_frames_test")
lst_image_paths = ['./static_frames_test/'+image for image in lst_images]

for i in lst_image_paths[0:-1]:
    img_cv = cv2.imread(i)
    cnts, img = preprocess_img(img_cv, crop=False)
    out_image = detect_forme(cnts, img)
    thresh = blur(out_image)
    new_cnts = get_contours(thresh)
    HuMo = get_humos(new_cnts)
    
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    msg = str(get_distance(HuMo).iloc[0].target) + ' distance: ' + str(round(get_distance(HuMo).iloc[0].distance,2))
    cv2.putText(thresh,
        msg,
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (50, 255, 50),  
        2,
        cv2.LINE_4)

    cv2.imshow('Image', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        