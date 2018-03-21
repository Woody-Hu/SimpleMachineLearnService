# -*- coding: utf-8 -*-
'''
Created on 2018年3月20日
@author: Administrator
'''

from pymongo import MongoClient
  
class MongoDbInterface(object):   
    
    def __init__(self):
        self.use_connection = MongoClient()
        self.change_db()
        self.change_collection()
        
    
    def change_db(self,input_db_name = "TestDb"):
        self.use_db = getattr(self.use_connection, input_db_name) 

    def change_collection(self,input_collection_name = "TestCollection"):
        if not input_collection_name is None:
            self.use_collection = getattr(self.use_db, input_collection_name)
       
    def insert_value(self,collection_name = None , **kw): 
        self.change_collection(collection_name)
        self.use_collection.insert(kw)

    def insert_many_value(self,input_values,collection_name = None):
        self.change_collection(collection_name)
        self.use_collection.insert_many(input_values)
        
    def update_value(self,where_value,update_values,collection_name = None):
        self.change_collection(collection_name)
        self.use_collection.update(where_value,{"$set":update_values},True)  

    def find_all(self):
        return self.use_collection.find()
    
    def find(self,collection_name = None , **kw):
        self.change_collection(collection_name)
        return self.use_collection.find(kw)   
    
    def reomve(self,collection_name = None , **kw):
        self.change_collection(collection_name)
        
        if not kw is None:
            self.remove(kw)
        else:
            self.remove()    
    
    def __del__(self):
        self.use_connection.close()
 

