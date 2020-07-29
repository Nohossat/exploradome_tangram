import numpy as np

def dist_humoment1(hu1,hu2):
    distance =  np.sum(abs(1/hu1-1/hu2))
    return distance

def dist_humoment2(hu1,hu2):
    distance =  np.sum(abs(hu1-hu2))
    return distance

def dist_humoment3(hu1,hu2):
    distance =  np.sum(abs(hu1-hu2)/abs(hu2))
    return distance
