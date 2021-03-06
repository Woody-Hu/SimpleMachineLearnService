# -*- coding: utf-8 -*-
'''
Created on 2018��3��21��

@author: Administrator
'''

from MongoDBInterface import MongoDbInterface

_use_name_collection ='modelNameAndJson'
    
_use_name_db = "machinelearndb"
    
_use_name_suffix_train_value ="trainValue"

_use_name_suffix_evaluate_value ="evaluateValue"

_str_use_json ="useJson"

_str_x_value="xvalue"

_str_y_value="yvalue"

_str_name='name'

_use_name_ckptfile_collection ="ckptfiles"

_use_name_metafile_collection ="ckptfiles"

_use_name_checkpointfile_collection ="checkpointfiles"

def model_name_check(if_need_model):
    def model_name_check_warp(func):
        def _model_name_check(*args,**kw):
            if args[0].if_contains_model_name(args[1]) and not if_need_model:
                return False
            elif not args[0].if_contains_model_name(args[1]) and if_need_model: 
                return False
            return func(*args,**kw)
        return _model_name_check
    return model_name_check_warp
   
class LearningModelService:
    
    def __init__(self):
        self._useDb = MongoDbInterface()
        self._useDb.change_db(_use_name_db)
        self._useDb.change_collection(_use_name_collection)
        pass
    
    def if_contains_model_name(self,input_name):
        if 0 != (self._useDb.find(name = input_name).count()):
            return True
        else:
            return False
   
    def get_all_names(self):
        find_vlaue = self._useDb.find_all()
        return_value = []
        
        for one_value in find_vlaue:
            return_value.append(one_value[_str_name])
        return return_value    
     
    @model_name_check(False)
    def insert_model(self,input_name,input_json):
        self._useDb.insert_value(name = input_name,useJson = input_json)
        return True
    
    @model_name_check(True)
    def insert_modelfile(self,input_name,input_ckpt_file,input_metafile,input_checkpointfile):
        self._useDb.del_file(_use_name_ckptfile_collection, input_name)
        self._useDb.del_file(_use_name_metafile_collection, input_name)
        self._useDb.del_file(_use_name_checkpointfile_collection, input_name)
        self._useDb.insert_file(_use_name_ckptfile_collection, input_name, input_ckpt_file)
        self._useDb.insert_file(_use_name_metafile_collection, input_name, input_metafile)
        self._useDb.insert_file(_use_name_checkpointfile_collection, input_name, input_checkpointfile)
    
    @model_name_check(True)
    def get_modelfile(self,input_name,input_ckpt_file,input_metafile,input_checkpointfile):  
        self._useDb.get_file(_use_name_ckptfile_collection, input_name, input_ckpt_file) 
        self._useDb.get_file(_use_name_metafile_collection, input_name, input_metafile) 
        self._useDb.get_file(_use_name_checkpointfile_collection, input_name, input_checkpointfile) 
        
    @model_name_check(True)
    def get_model_json(self,input_name):
        return self._useDb.find(name = input_name)[0][_str_use_json]
    
    @model_name_check(True)
    def insert_train_value(self,input_name,input_x_values,input_y_values):
        self._insert_model_result_value(input_name + _use_name_suffix_train_value,input_x_values,input_y_values)
    
    @model_name_check(True)
    def insert_evaluate_value(self,input_name,input_x_values,input_y_values):
        self._insert_model_result_value(input_name + _use_name_suffix_evaluate_value,input_x_values,input_y_values)
    
    @model_name_check(True)
    def get_train_value(self,input_name,input_limit = None,input_skip = None):
        return self._get_model_result_value(input_name + _use_name_suffix_train_value,input_limit,input_skip)
    
    @model_name_check(True)
    def get_evaluate_value(self,input_name,input_limit = None,input_skip = None):
        return self._get_model_result_value(input_name + _use_name_suffix_evaluate_value,input_limit,input_skip)
    
    def _get_model_result_value(self,input_collection_name,input_limit,input_skip):
        self._useDb.change_collection(input_collection_name)   
        findValues = self._useDb.find_all(input_limit, input_limit)
        self._useDb.change_collection(_use_name_collection)
        
        x_value =[]
        y_value =[]
        
        for oneValue in findValues:
            x_value.append(oneValue[_str_x_value])
            y_value.append(oneValue[_str_y_value])
        return (x_value,y_value)
    
    def _insert_model_result_value(self,input_collection_name,input_x_values,input_y_values):
        if len(input_x_values) != len(input_y_values):
            return False
        values = []
        
        for index,value in enumerate(input_x_values):
            temp_dict = {_str_x_value:value,_str_y_value:input_y_values[index]}
            values.append(temp_dict)
            
        self._useDb.insert_many_value(values, input_collection_name)
        #当前表回置
        self._useDb.change_collection(_use_name_collection)   


