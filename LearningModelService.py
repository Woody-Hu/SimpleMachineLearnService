# -*- coding: utf-8 -*-
'''
Created on 2018��3��21��

@author: Administrator
'''

from MongoDBInterface import MongoDbInterface

_use_name_collection ='modelNameAndJson'
    
_use_name_db = "machinelearndb"
    
_use_name_suffix_train_value ="trainValue"

_str_use_json ="useJson"

   

class LearningModelService:
    
    def __init__(self):
        self._useDb = MongoDbInterface()
        self._useDb.change_db(_use_name_db)
        self._useDb.change_collection(_use_name_collection)
        pass
    
    def if_contains_model_name(self,input_name):
        if 0 != (self._useDb.find(name = input_name).retrieved):
            return True
        else:
            return False
    
    def model_name_check(self,if_need_model):
        def model_name_check_warp(func):
            def _model_name_check(*args,**kw):
                if self.if_contains_model_name(args[1]) and not if_need_model:
                    return False
                    
                
        
        
    def get_all_names(self):
        find_vlaue = self._useDb.find_all()
        return_value = []
        
        for one_value in find_vlaue:
            return_value.append(one_value)
        return return_value    
       
    def insert_model(self,input_name,input_json):
        if self.if_contains_model_name(input_name):
            return False
        self._useDb.insert_value(name = input_name,useJson = input_json)
        return True
    
    def get_model_json(self,input_name):
        if not self.if_contains_model_name(input_name):
            return False
        return self._useDb.find(name = input_name)[0][_str_use_json]
    
    def insert_model_result_value(self,input_name,input_x_values,input_y_values):
        if not self.if_contains_model_name(input_name):
            return False
        
    