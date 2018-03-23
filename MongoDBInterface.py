# -*- coding: utf-8 -*-
'''
Created on 2018年3月20日
@author: Administrator
'''

from pymongo import MongoClient
from gridfs import GridFS
  
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

    def find_all(self,limit_value = None,skip_value = None):
        
        temp_value = self.use_collection.find();
        if not limit_value is None and not skip_value is None:
            return self.use_collection.find().limit(limit_value)
        else:
            return temp_value
        
    def find(self,collection_name = None,limit_value = None,skip_value = None, **kw):
        self.change_collection(collection_name)
        temp_value = self.use_collection.find(kw)
        if not limit_value is None and not skip_value is None:
            return self.use_collection.find().limit(limit_value)
        else:
            return temp_value 
    
    def insert_file(self,collection_name,useFilename,input_file_path):
        fs = GridFS(self.use_db, collection_name)  
        fstream = open(input_file_path.decode('utf-8'),'rb')
        data = fs.read()
        return fstream.put(data,file_name = useFilename)
     
    def get_file(self,collection_name,useFilename,use_file_path):
        fs = GridFS(self.use_db, collection_name)  
        fileInfo = fs.get_last_version(useFilename)
        data = fileInfo.read()
        fstream = open(use_file_path.decode('utf-8'),'wb')
        fstream.write(data)
        fstream.close()
        return None
     
    def del_file(self,collection_name,useFilename):  
        fs = GridFS(self.use_db, collection_name)
        fileInfo = fs.get_last_version(useFilename)
        fs.delete(fileInfo._id)
        return None
    
    def reomve(self,collection_name = None , **kw):
        self.change_collection(collection_name)
        
        if not kw is None:
            self.remove(kw)
        else:
            self.remove()    
    
    def __del__(self):
        self.use_connection.close()
 

