# -*- coding: utf-8 -*-
'''
Created on 2018年3月20日

@author: Administrator
'''

from pymongo import MongoClient
from Singletondecorator import *

@singleton
class MongoDbInterface(object):   
    
    def __init__(self):
        self.use_connection = MongoClient()
        self.change_db()
        self.change_collection()
        
    
    def change_db(self,input_db_name = "TestDb"):
        self.use_db = getattr(self.use_connection, input_db_name) 

    def change_collection(self,input_collection_name = "TestCollection"):
        self.use_collection = getattr(self.use_db, input_collection_name)

    def insert_value(self,collection_name = None , **kw):
        
        if not collection_name is None:
            self.change_collection(collection_name)
        self.use_collection.insert(kw)
        
    
    def find_all(self):
        return self.use_collection.find()
    
    def find(self,collection_name = None , **kw):
        if not collection_name is None:
            self.change_collection(collection_name)
        return self.use_collection.find(kw)   
    
    def __del__(self):
        self.use_connection.close()
        
    
    





 