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
    return tf.Variable(tf.zeros(shpae[1], dtype = tf.float64, name = useName))

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
    return tf.matmul(inputValue, inputWeight) + inputBiases

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

def prepare_palceholder(inputShapeDescribe):
    xShape = inputShapeDescribe[0]
    
    yShape = inputShapeDescribe[-1]
    
    x = tf.placeholder(tf.float64, [1,xShape])
    
    y_ = tf.placeholder(tf.float64,[1,yShape])
    
    return x,y_

inputDescrib = [2,3,1]

x,y_ = prepare_palceholder(inputDescrib)

useShapes = make_shape(inputDescrib)

print(useShapes)

weights = []



biases = []



for index,val in enumerate(useShapes):
    weights.append(make_variable(val,useName = 'weight' + str(index)))
    biases.append(make_biases(val,useName = 'biases' + str(index)))
    

firstLayerValue = makeLayer_calculate(x,weights[0],biases[0])

print(firstLayerValue)

y = makeLayer_calculate(firstLayerValue,weights[1],biases[1])

print(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(weights[0]))
    print(sess.run(weights[1]))
    
    print(sess.run(biases[0]))
    print(sess.run(biases[1]))
    
    resultValue = sess.run(firstLayerValue,feed_dict={x:[[1,1]],y_:[[1]]})
    print(resultValue)
    resultValue

    

    
    
    


    
    