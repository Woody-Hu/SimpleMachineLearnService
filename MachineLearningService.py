# -*- coding: utf-8 -*-
'''
Created on 2018��2��23��

@author: Administrator
'''

import tensorflow as tf
from _overlapped import NULL

# 制作变量
def MakeVariable(shpae,useName,useStddev = 1):
    return tf.Variable(tf.random_normal(shpae,dtype = tf.float64, stddev = useStddev, name = useName))

def MakeBiases(shpae,useName):
    return tf.Variable(tf.zeros(shpae, dtype = tf.float64, name = useName))


def MakeShape(inputShapeDescribe):
    return NULL

def MakeLayerCalculate(inputValue,inputWeight,inputBiases):
    return tf.multiply(inputValue, inputWeight) + inputBiases

def MakeLayerActive(inputValue,inputKind):
    if 0 == inputKind:
        return inputValue
    elif 1 == inputKind:
        return tf.nn.relu(inputValue)
    elif 2 == inputKind:
        return tf.nn.tanh(inputValue)
    else:
        return tf.nn.sigmoid(inputValue)
    
    
    
    


    
    