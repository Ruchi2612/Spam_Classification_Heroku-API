# -*- coding: utf-8 -*-
"""
Created on Sun May 23 06:12:07 2021

@author: Ruchilekha
"""

# 1. Import Libraries
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

# 2. Load the model from disk
app = Flask(__name__)

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename,'rb')) 
cv = pickle.load(open('transform.pkl','rb')) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug=True)