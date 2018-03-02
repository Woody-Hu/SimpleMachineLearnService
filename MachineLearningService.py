# -*- coding: utf-8 -*-
'''
Created on 2018��2��23��

@author: Administrator
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from LayerDescribeBean import *

from MatrixDescribeBean import MatrixDescribeClass

from collections import Iterable

from ModelMakeRequestBean import ModelMakeRequestBean

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np


def make_variable(shpae,useName,useStddev = 0.1):
    '''
    制作变量
    '''
   
    return tf.Variable(tf.random_normal(shpae,dtype = tf.float32, stddev = useStddev, name = useName))
  
        

def make_biases(shpae,useName):
    '''
    制作偏移量
    '''
    return tf.Variable(tf.zeros(shpae[-1], dtype = tf.float32, name = useName) + 0.1)

def make_shape(inputShapeDescribe):
    '''
    根据描述封装制作矩阵
    '''
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
    '''
    进行一层神经网络计算
    '''
    return tf.matmul(inputValue, inputWeight) + inputBiases

def make_layer_active(inputValue,inputKind):
    '''
    层激活函数
    '''
    if 0 == inputKind:
        return inputValue
    elif 1 == inputKind:
        return tf.nn.relu(inputValue)
    elif 2 == inputKind:
        return tf.nn.tanh(inputValue)
    else:
        return tf.nn.sigmoid(inputValue)

def prepare_input_output_palceholder(inputShapeDescribe):
    '''
    根据描述制作输入输出占位
    '''
    xShape = inputShapeDescribe[0].shape
    
    yShape = inputShapeDescribe[-1].shape
    
    x = make_one_placeholder(xShape)
    
    y_ = make_one_placeholder(yShape)
    
    return x,y_

def make_one_placeholder(inputShape):
    temp_lst =[]
    temp_lst.append(None)
    
    if isinstance(inputShape, int):
        temp_lst.append(inputShape)
    elif isinstance(inputShape, Iterable):
        for one_shape in inputShape:
            temp_lst.append(one_shape)
            
    temp_placeholder = tf.placeholder(tf.float32,shape = tuple(temp_lst))
    return temp_placeholder;
    
def nn_forward_caculate(inputShapeDescribe,if_get_y_ = False):
    '''
    全连接神经网络向前计算接口
    '''
    
    x,y_ = prepare_input_output_palceholder(inputShapeDescribe)
    
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
    '''
    输入检查
    '''
    for val in inputShapeDescribe:
        if not isinstance(val, BaseLayerDescribeBean):
            return False
    return True   

def nn_prediction(inputShapeDescribe,inputX):
    if not inputCheck(inputShapeDescribe):
        raise Exception()
    
    (result,x) = prepare_input_output_palceholder(inputShapeDescribe)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        resultValue = sess.run(result,feed_dict = {x:inputX})

    return resultValue

def train(inputShapeDescribe,inputX,inputY_,input_step = 5000,inputBatchSize = 8, use_save_path = None):
    if not inputCheck(inputShapeDescribe):
        raise Exception()
    
    (layer_calculate_result,x,y_) = nn_forward_caculate(inputShapeDescribe,True)
    
    (tranin_step,cross_entropy) = back_propagation(layer_calculate_result,y_)
    
    input_data_size = len(inputY_)
    
    returnValue = []
    
    if use_save_path != None:
        saver = tf.train.Saver();
    
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
        
        if use_save_path != None:
            saver.save(sess, use_save_path)
                            
    return returnValue

def get_variable_lastShape(input_variable):
    '''
    获得输入变量的最后一阶的形状描述
    '''
    return input_variable.get_shape().as_list()[-1]

def cnn_layer_calculate(input_value,input_layer_name,input_conv_matrix_bean,
                        input_pool_matrix_bean,input_active_kind = 1):
    '''
    一层卷积+最大池化计算
    '''
    last_deep = get_variable_lastShape(input_value)
    
    #制作卷积层权重形状矩阵
    use_conv_weight = [input_conv_matrix_bean.get_shape[0],input_conv_matrix_bean.get_shape[1]
                      ,last_deep,input_conv_matrix_bean.get_shape[2]]
    #制作卷积层偏移形状矩阵
    use_conve_biases = [input_conv_matrix_bean.get_shape[2]]
    #制作卷积层步长矩阵
    use_conv_strides = [1,input_conv_matrix_bean.get_strides[0],input_conv_matrix_bean.get_strides[1],1]
    #池化层的形状矩阵
    use_pool_shape = [1,input_pool_matrix_bean.get_shape[0],input_pool_matrix_bean.get_shape[1],1]
    #池化层的步长矩阵
    use_pool_strides = [1,input_pool_matrix_bean.get_strides[0],input_pool_matrix_bean.get_strides[1],1]
    
    
    with tf.variable_scope(input_layer_name):
        weight_variable = make_variable(use_conv_weight,"weight")
        biases_variable = make_biases(use_conve_biases, "biases")
        #卷积计算
        conve_result = tf.nn.conv2d(input_value,weight_variable,strides= use_conv_strides,padding= 'VALID')
        result = tf.nn.bias_add(conve_result, biases_variable)
        #激活
        result = make_layer_active(result, input_active_kind)
        pool_result = tf.nn.max_pool(result, use_pool_shape, use_pool_strides, 'VALID')
        return pool_result 
    
    return None
    
def define_let_net_5_calculate(input_request_bean,input_batch):
     
    #声明第一层卷积层形状定义 （5,5,32） 步长 （1,1,1,1）
    conv_matrix_bean_one = MatrixDescribeClass(MatrixDescribeClass.get_default_conv_shape(32),MatrixDescribeClass.get_default_strides())
    
    pool_matrix_bean = MatrixDescribeClass(MatrixDescribeClass.get_default_pool_shape(),MatrixDescribeClass.get_default_strides())
    
    conv_layer_one = CNNPoolLayerDescriBean(conv_matrix_bean_one,pool_matrix_bean)
    
    #声明第三层卷积层形状定义 （5,5,64） 步长 （1,1,1,1）
    conv_matrix_bean_two = MatrixDescribeClass(MatrixDescribeClass.get_default_conv_shape(64),MatrixDescribeClass.get_default_strides())
    
    conv_layer_two = CNNPoolLayerDescriBean(conv_matrix_bean_two,pool_matrix_bean)

    #第五层全连接层
    nn_layer_one = NNLayerDescribeBean(512,1)
    
    #第六层全连接层
    nn_layer_two = NNLayerDescribeBean(input_request_bean.all_bean[-1].shape,0)
    
    all_bean = [conv_layer_one,conv_layer_two,nn_layer_one,nn_layer_two]
    
    use_request = ModelMakeRequestBean(all_bean,input_request_bean.x_shape)
    
    return define_cnn_calculate(use_request,input_batch)
   
def define_cnn_calculate(input_request_bean,input_batch):
    '''
    定义一个卷积神经元网络计算
    '''
    x = make_one_placeholder(input_request_bean.x_shape)
    
    temp_result = x
    
    has_reShape = False
    
    last_nn_shape = 0
    
    #循环建立网络
    for index,one_bean in enumerate(input_request_bean.all_bean):
        if isinstance(one_bean, CNNPoolLayerDescriBean) and (not has_reShape):
            #卷积+池化计算
            temp_result = cnn_layer_calculate(temp_result,"cnn_pool_layer" + str(index),one_bean.conv_shape_bean,one_bean.pool_shape_bean)
        #当开始全连接层时
        if isinstance(one_bean, NNLayerDescribeBean):
            #最初时开始调整Shape
            if not has_reShape:
                #重置标签
                has_reShape =  True
                result_shape = temp_result.get_shape().as_list()
                #获取长、宽、深
                all_rank = result_shape[1]*result_shape[2]*result_shape[3]
                #shape合并
                temp_result = tf.reshape(temp_result, [input_batch,all_rank])
                #设置nn层形状传递
                last_nn_shape = all_rank
            #形成nn层形状 权值矩阵 偏移值矩阵      
            temp_shape = [last_nn_shape,one_bean.shape]
            temp_weight = make_variable(temp_shape,"nn_weight_layer" + str(index))
            temp_basies = make_biases(temp_shape, "nn_basies_layer" + str(index))
            
            #全连接层计算
            temp_result = make_layer_calculate(temp_result,temp_weight,temp_basies)
            temp_result = make_layer_active(temp_result,one_bean.active_kind)
            #设置nn层形状传递
            last_nn_shape = temp_shape[1]
            
    return x,temp_result

def test_x_reshape(input_batch_size,input_x):
    return np.reshape(input_x,[input_batch_size,28,28,1])

minist = input_data.read_data_sets("/path/to/MINIST_data/",one_hot= True)


use_x_shape =(28,28,1)
use_y_shape = 10

use_nn_layer_shape = NNLayerDescribeBean(use_y_shape)

use_request_bean = ModelMakeRequestBean([use_nn_layer_shape,],use_x_shape)

batch_size = 500

(x,y) = define_let_net_5_calculate(use_request_bean,batch_size)

y_ = make_one_placeholder(use_y_shape)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_, 1))

cross_entropy_mean = tf.reduce_mean(cross_entropy)

correct_p = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_p,tf.float32))

tranin_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_mean)



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for i in range(30000):
        xs,ys = minist.train.next_batch(batch_size)
        n_xs = test_x_reshape(batch_size,xs)
        
        y_result,loss,step = sess.run((y,cross_entropy_mean,tranin_step),feed_dict ={x:n_xs,y_:ys })
        
        if i % 1 == 0:
            xe,ye = minist.validation.next_batch(batch_size)
            xe = test_x_reshape(batch_size, xe)
            acc = sess.run(accuracy,feed_dict = {x:xe,y_:ye})
            print(loss)
            print(acc)
            print("")
        
        

       

  
    