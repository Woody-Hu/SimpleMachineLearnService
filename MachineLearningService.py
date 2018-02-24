# -*- coding: utf-8 -*-
'''
Created on 2018��2��23��

@author: Administrator
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from ShapeDescribeBean import ShapeDescribeClass


def make_variable(shpae,useName,useStddev = 1):
    return tf.Variable(tf.random_normal(shpae,dtype = tf.float64, stddev = useStddev, name = useName))

def make_biases(shpae,useName):
    return tf.Variable(tf.zeros(shpae[1], dtype = tf.float64, name = useName))

def make_shape(inputShapeDescribe):
    returnValue = []
    lastValue = 0
    for index, val in enumerate(inputShapeDescribe):
        if 0 == index:
            lastValue = val.shape
            continue
        tempValue = [lastValue,val.shape]
        returnValue.append(tempValue)
        lastValue = val.shape
        
    return returnValue

def make_layer_calculate(inputValue,inputWeight,inputBiases):
    return tf.matmul(inputValue, inputWeight) + inputBiases

def make_layer_active(inputValue,inputKind):
    if 0 == inputKind:
        return inputValue
    elif 1 == inputKind:
        return tf.nn.relu(inputValue)
    elif 2 == inputKind:
        return tf.nn.tanh(inputValue)
    else:
        return tf.nn.sigmoid(inputValue)

def prepare_palceholder(inputShapeDescribe):
    xShape = inputShapeDescribe[0].shape
    
    yShape = inputShapeDescribe[-1].shape
    
    x = tf.placeholder(tf.float64, [1,xShape])
    
    y_ = tf.placeholder(tf.float64,[1,yShape])
    
    return x,y_

def forward_caculate(inputShapeDescribe,inputX,inputY):
    
    x,y_ = prepare_palceholder(inputShapeDescribe)
    
    useShapes = make_shape(inputShapeDescribe)
    
    weights = []
    biases = []

    base_weight_str = 'weight'
    base_biases_str ="biases"
    

    for index,val in enumerate(useShapes):
        weights.append(make_variable(val,useName = base_weight_str + str(index)))
        biases.append(make_biases(val,useName = base_biases_str + str(index)))
    
    layer_calculate_result = x
    
    for index,oneDes in enumerate(inputShapeDescribe):
        if 0 == index:
            continue
        layer_calculate_result = make_layer_calculate(layer_calculate_result, weights[index - 1], biases[index - 1])
        layer_calculate_result = make_layer_active(layer_calculate_result,oneDes.active_kind)
     
     
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        resultValue = sess.run(layer_calculate_result,feed_dict={x:[inputX],y_:[inputY]})


    return resultValue



inputDescrib = [ShapeDescribeClass(2),ShapeDescribeClass(1,2)]

returnValue = forward_caculate(inputDescrib,[1,1],[1])

print(returnValue)

    

    
    
    


    
    