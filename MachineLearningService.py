# -*- coding: utf-8 -*-
'''
Created on 2018��2��23��

@author: Administrator
'''

import tensorflow as tf


# 制作变量
def make_variable(shpae,useName,useStddev = 1):
    return tf.Variable(tf.random_normal(shpae,dtype = tf.float64, stddev = useStddev, name = useName))

# 制作偏移值
def make_biases(shpae,useName):
    return tf.Variable(tf.zeros(shpae, dtype = tf.float64, name = useName))

# 根据输入描述制作shape
def make_shape(inputShapeDescribe):
    returnValue = []
    lastValue = 0
    for index, val in enumerate(inputShapeDescribe):
        if 0 == index:
            lastValue = val
            continue
        tempValue = [lastValue,val]
        returnValue.append(tempValue)
        lastValue = val
        
    return returnValue

# 层级计算
def makeLayer_calculate(inputValue,inputWeight,inputBiases):
    return tf.multiply(inputValue, inputWeight) + inputBiases

# 结果激活
def make_layeractive(inputValue,inputKind):
    if 0 == inputKind:
        return inputValue
    elif 1 == inputKind:
        return tf.nn.relu(inputValue)
    elif 2 == inputKind:
        return tf.nn.tanh(inputValue)
    else:
        return tf.nn.sigmoid(inputValue)

def forward_calculate(inputShapeDescribe,inputXValue):
    return None


    
    
    


    
    