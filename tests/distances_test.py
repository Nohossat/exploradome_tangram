
"""
Created on Tue Jul 28 15:11:06 2020

@author: Renata
"""
import pytest
import distances.py

#test of distance humoment1
def dist_humoment1_test():
    assert dist_humoment1(0.026322267294741648, 0.0027161374010481088) == 330.179238664131

#test of distance humoment2
def dist_humoment2_test():
    assert dist_humoment2(0.026322267294741648, 0.0027161374010481088) == 0.0236061298936935

#test of distance humoment3
def dist_humoment3_test():
    assert dist_humoment3(0.026322267294741648, 0.0027161374010481088) == 8.69106617529156
    #message erreur, commentaire