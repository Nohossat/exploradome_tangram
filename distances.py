import numpy as np


def dist_humoment1(hu1,hu2):
    """
    return sum absolute difference of inverse of humoment value
    author: @Gautier
    """
    distance =  np.sum(abs(1/hu1-1/hu2))
    return distance


def dist_humoment2(hu1,hu2):
    """
    return the sum of manatan distance, sum of absolute difference
    author: @Gautier
    """
    distance =  np.sum(abs(hu1-hu2))
    return distance

def dist_humoment3(hu1,hu2):
    """
    return sum of absolute difference  divide absolute of second humoment
    author: @Gautier
    """
    distance =  np.sum(abs(hu1-hu2)/abs(hu2))
    return distance

def dist_humoment4(hu1,hu2):
    """
    return sum of euclidienne distance,sum of squart of pow difference
    author: @Gautier
    """
    distance = np.linalg.norm(hu1-hu2)
    return distance

