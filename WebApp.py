# -*- coding: utf-8 -*-
'''
Created on 2018��3��15��

@author: Administrator
'''

from flask import Flask
from flask import request

app = Flask(__name__)    

@app.route('/')           
def index():
    return 'Home' 

@app.route('/addmodel')   
def add_model():
    return 'Add modle';

if __name__ == '__main__':
    app.run(debug=True)          