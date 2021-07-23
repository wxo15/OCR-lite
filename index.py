from flask import Flask
import os
from flask import render_template, flash, request, redirect, url_for
from joblib import dump, load
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='')

@app.route("/", methods=['GET', 'POST']) 
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    else: #request.method == 'POST':
        if request.json['image'] is None:
            flash('No file part')
            return redirect(request.url)
        img = np.array(request.json['image'])
        myModel = app.config['MODEL']
        scaler = StandardScaler()
        scaler.mean_ = app.config['MEAN']
        scaler.var_ = app.config['VAR']
        scaler.scale_ = app.config['SCALE']
        img = scaler.transform(img.reshape(1, -1))
        print(img)
        res = myModel.predict(img)
        print(res)
        return render_template('index.html',result=res)    

def load_model_from_file():
    myModel = load('logres.m5')
    myMean = np.load('logres_mean.npy')
    myVar = np.load('logres_var.npy')
    myScale = np.load('logres_scale.npy')
    return (myModel, myMean, myVar, myScale)

def main():
    (myModel, myMean, myVar, myScale) = load_model_from_file()
    app.config['MODEL'] = myModel
    app.config['MEAN'] = myMean
    app.config['VAR'] = myVar
    app.config['SCALE'] = myScale
    app.run()

main()