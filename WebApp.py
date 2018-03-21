# -*- coding: utf-8 -*-
'''
Created on 2018��3��15��

@author: Administrator
'''

from flask import Flask
from flask import request
from LayerDescribeBean import *
from MatrixDescribeBean import MatrixDescribeBean
from ModelMakeRequestBean import ModelMakeRequestBean

import json

app = Flask(__name__)    

@app.route('/')           
def index():
    return 'Home' 

@app.route('/addmodel')   
def add_model():
    return 'Add modle';

@app.route('/checkmodelname/<string:name>')
def check_model_name(name): 
    if name == '1':
        return 'A'
    else:
        return 'B'
    
def json_transform(strJ):
    #获取Json字典
    dictJson = json.loads(strJ)
    
    isCNN = dictJson['isCNN'] == str(True)
    
    isUseLetNet5 = False
    
    if isCNN:
        isUseLetNet5 = dictJson['isUseLetNet5'] == str(True)
    
    input_shape = dictJson['input_shape']
    
    output_shape = dictJson['output_shape']
    
    #输出层全连接层
    output_nnLayer = NNLayerDescribeBean(output_shape)
    
    inner_shape = dictJson['inner_shape']
    
    use_shapes = []
    
    for oneInnerShape in inner_shape:
        
        isCNNLayer = oneInnerShape['isCNNLayer']== str(True)
        use_active_kind = oneInnerShape['useActiveKind']
        add_layer = None
        
        if not isCNNLayer:
            use_shape = oneInnerShape['use_shape']
            add_layer = NNLayerDescribeBean(use_shape,use_active_kind)
            
        else:
            use_cnn_shape = oneInnerShape['use_cnn_shape']
            use_cnn_strids = oneInnerShape['use_cnn_strids']
            use_pool_shape = oneInnerShape['use_pool_shape']
            use_pool_strids = oneInnerShape['use_pool_strids']
            add_layer = CNNPoolLayerDescriBean(MatrixDescribeBean(use_cnn_shape,use_cnn_strids),\
                                               MatrixDescribeBean(use_pool_shape,use_pool_strids))
       
        use_shapes.append(add_layer)
    
    #添加输出层
    use_shapes.append(output_nnLayer)
    
    if isUseLetNet5:
        return ModelMakeRequestBean([output_nnLayer,],input_shape,True)
    else:
        return ModelMakeRequestBean(use_shapes,input_shape)


if __name__ == '__main__':
    #test_str = '{"isCNN":true,"isUseLetNet5":false,"input_shape":[10],"output_shape":[1],"inner_shape":[{"isCNNLayer":false,"useActiveKind":0,"use_shape":[5]}]}'
    #data = json_transform(test_str)
    print("test")
