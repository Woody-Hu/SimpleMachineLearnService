'''
Created on 2018��3��1��

@author: Administrator
'''

from collections import Iterable
from LayerDescribeBean import *

class ModelMakeRequestBean(object):
    '''
    classdocs
    '''
    
    _all_layer_desctibe_bean = []
    
    _x_shape = []
    
    _y_shape = []
    
    _if_use_cnn = False
    
    _if_use_net_5 = False
    
    def __init__(self, input_all_layer_desctibe_bean,input_x,input_y_ ,input_if_use_net_5 = False):
        '''
        Constructor
        '''
        self._all_layer_desctibe_bean = input_all_layer_desctibe_bean
        self._x_shape = input_x
        self._y_shape= input_y_
        self._if_use_net_5 = input_if_use_net_5
        self._inputCheck()
    
    def _inputCheck(self):
        
        if not isinstance(self._all_layer_desctibe_bean, Iterable) or not isinstance(self._x, Iterable)or not isinstance(self._y_shape, Iterable):
            raise Exception()
        
        if_found_nn = False
        
        #������
        for one_des_bean in self._all_layer_desctibe_bean:
            #����Ƿ��������
            if not isinstance(one_des_bean, BaseLayerDescribeBean):
                raise Exception()
            #ֻҪ��cnn��װ����Ϊcnnģʽ
            elif (not self._if_use_cnn) and isinstance(one_des_bean, CNNPoolLayerDescriBean):
                self._if_use_cnn = True
            #����ȫ���Ӳ���ж�   
            elif isinstance(one_des_bean, NNLayerDescribeBean) and (not if_found_nn):
                if_found_nn = True  
            elif if_found_nn and (not isinstance(one_des_bean, NNLayerDescribeBean)):
                raise Exception()
                        
        return  
    
    @property
    def x_shape(self):
        return self._x_shape
    
    @property
    def y_shape(self):
        return self._y_shape
    
    @property
    def if_use_cnn(self):
        return self._if_use_cnn
    
    @property
    def if_use_net_5(self):
        return self._if_use_net_5