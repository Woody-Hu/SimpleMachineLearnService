# -*- coding: utf-8 -*-
'''
Created on 2018��3��20��

@author: Administrator
'''

def singleton(cls, *args, **kw):
    _instance_dict = {}
    
    def _singleton_inner():
        if cls not in  _instance_dict:
            _instance_dict[cls] = cls(*args,**kw)
        return _instance_dict[cls]
    
    return _singleton_inner

