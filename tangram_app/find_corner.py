import cv2
import numpy as np


def get_nb_corners(img):
    # img = cv2.imread()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,5,3,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    """
    This part is not for production
    It allow to visualize on an image the corner point in red
    """
    #for i in range(1, len(corners)):
        # print(corners[i])
        # print(len(corners))
        # img[dst>0.1*dst.max()]=[0,0,255]
        # cv2_imshow(img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
        
    return len(corners)



"""
Results nb of corners
bol = 12
####
coeur = 13
marteau = 13
montagne = 13
####
bateau = 14
maison = 14
tortue = 14
####
chat = 15 
cygne = 15
pont = 15 
####
lapin = 19
####
renard = 20 
"""