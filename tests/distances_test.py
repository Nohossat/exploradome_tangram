from .distances import *
import pytest

def dist_humoment1_test():
    assert dist_humoment1(0.026322267294741648, 0.0027161374010481088) == 330.179238664131

def dist_humoment2_test():
    assert dist_humoment2(0.026322267294741648, 0.0027161374010481088) == 0.0236061298936935

def dist_humoment3_test():
    assert dist_humoment3(0.026322267294741648, 0.0027161374010481088) == 8.69106617529156
    #message erreur, commentaire