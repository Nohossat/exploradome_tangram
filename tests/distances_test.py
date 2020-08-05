
"""
Created on Tue Jul 28 15:11:06 2020

@author: Renata
"""
import pytest

from tangram_app.distances import *

#test of distance humoment1
def test_dist_humoment1():
    assert dist_humoment1(0.026322267294741648, 0.0027161374010481088) == 330.17923866413116, 'distance_humoment1 not working'

#test of distance humoment2
def test_dist_humoment2():
    assert dist_humoment2(0.026322267294741648, 0.0027161374010481088) == 0.02360612989369354, 'distance_humoment2 not working'

#test of distance humoment3
def test_dist_humoment3():
    assert dist_humoment3(0.026322267294741648, 0.0027161374010481088) == 8.691066175291558, 'distance_humoment3 not working'
    #message erreur, commentaire
