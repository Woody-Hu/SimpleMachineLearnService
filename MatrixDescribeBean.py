# -*- coding: utf-8 -*-

'''
Created on 2018��2��28��

@author: Administrator
'''

class MatrixDescribeBean(object):
    '''
    classdocs
    '''
    _this_shape = None
    
    _this_strides = None


    def __init__(self, input_shape,input_strides):
        '''
        Constructor
        '''
        self._this_shape = input_shape
        self._this_strides = input_strides
   
    @property
    def get_shape(self):
        return self._this_shape   
    
    @property
    def get_strides(self):
        return self._this_strides
    
    @classmethod
    def get_default_conv_shape(cls,input_deep):
        return (5,5,input_deep)
    
    @classmethod
    def get_default_pool_shape(cls):
        return (2,2)
    
    @classmethod
    def get_default_strides(cls):
        return (1,1)