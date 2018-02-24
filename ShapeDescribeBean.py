# -*- coding: utf-8 -*-
'''
Created on 2018��2��24��

@author: Administrator
'''

class ShapeDescribeClass(object):

    '''
    classdocs
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
        self._activeKind = inputActiveKind
    
    @property
    def shape(self):
        return self._shape    
    
    @property
    def active_kind(self):
        return self._activeKind

 
    