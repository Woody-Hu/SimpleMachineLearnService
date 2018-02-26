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
    
    x = tf.placeholder(tf.float64, shape = (None,xShape))
    
    y_ = tf.placeholder(tf.float64,shape = (None,yShape))
    
    return x,y_

def forward_caculate(inputShapeDescribe,if_get_y_ = False):
    
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
     
    if if_get_y_:
        return layer_calculate_result,x,y_
    else:
        return layer_calculate_result,x
    
def back_propagation(inputResult,inputY_):
    
    cross_entropy = -tf.reduce_mean(inputY_*tf.log(inputResult)) 
    learning_reat = 0.001
    tranin_step = tf.train.AdamOptimizer(learning_reat).minimize(cross_entropy)
    
    return tranin_step,cross_entropy

def inputCheck(inputShapeDescribe):
    for val in inputShapeDescribe:
        if not isinstance(val, ShapeDescribeClass):
            return False
    return True   


def prediction(inputShapeDescribe,inputX):
    if not inputCheck(inputShapeDescribe):
        raise Exception()
    
    (result,x) = forward_caculate(inputShapeDescribe)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        resultValue = sess.run(result,feed_dict = {x:inputX})

    return resultValue

def train(inputShapeDescribe,inputX,inputY_,input_step = 5000,inputBatchSize = 8):
    if not inputCheck(inputShapeDescribe):
        raise Exception()
    
    (layer_calculate_result,x,y_) = forward_caculate(inputShapeDescribe,True)
    
    (tranin_step,cross_entropy) = back_propagation(layer_calculate_result,y_)
    
    input_data_size = len(inputY_)
    
    returnValue = []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(input_step):
            start = i*inputBatchSize
            end = min(start + inputBatchSize, input_data_size)
            if start > end:
                start = end
            sess.run(tranin_step,feed_dict = {x:inputX[start:end],y_:inputY_[start:end]})  
        
        if i % 1000 == 0:
            tempValue = sess.run(cross_entropy,feed_dict = {x:inputX,y_:inputY_})
            returnValue.append('after %d steps cross_entropy is %g'%(i,tempValue))           
    return returnValue

    
a = [ShapeDescribeClass(2),ShapeDescribeClass(1)]

b = prediction(a,[[1,1],[2,2]])

print(b)
    
    
    


    
    