# -*- coding: utf-8 -*-
'''
Created on 2018��2��24��

@author: Administrator
'''

class BaseLayerDescribeBean(object):
    '''
    计算层描述基础类
    '''
    #激活形式
    _activeKind =0
    
    def __init__(self,inputActiveKind):
        '''
        Constructor
        '''
        self._activeKind = inputActiveKind
        
     
    @property
    def active_kind(self):
        return self._activeKind    

class NNLayerDescribeBean(BaseLayerDescribeBean):

    '''
    全连接层描述类
    '''
    #shape描述
    _shape = 1
    
    #激活形式
    _activeKind =0

    def __init__(self, inputShape,inputActiveKind = 0):
        '''
        Constructor
        '''
        self._shape = inputShape
        super(NNLayerDescribeBean,self).__init__(inputActiveKind)
    
    @property
    def shape(self):
        return self._shape    
    
    @property
    def active_kind(self):
        return self._activeKind


class CNNPoolLayerDescriBean(BaseLayerDescribeBean):
    '''
   卷积+池化层描述类
    '''
    _conv_shape = None
    
    _pool_shape = None
    
    def __init__(self,input_conve_bean,input_pool_bean,inputActiveKind = 1):
        '''
        Constructor
        '''
        self._conv_shape = input_conve_bean
        self._pool_shape = input_pool_bean
        super(CNNPoolLayerDescriBean,self).__init__(inputActiveKind)
    
    @property
    def conv_shape_bean(self):
        return self._conv_shape
    
    @property
    def pool_shape_bean(self):
        return self._pool_shape