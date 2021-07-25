from flask import Flask
import os
from flask import render_template, flash, request, redirect, url_for
from joblib import dump, load
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='')

@app.route("/", methods=['GET']) 
def render():
    return render_template('index.html', modelName=app.config['MODELNAME'])

@app.route("/predict", methods=['POST']) 
def predict():
    if request.json['image'] is None:
        flash('No file part')
        return redirect(request.url)
    # print(request.json['image'])
    img = np.array(request.json['image'])
    myModel = app.config['MODEL']
    scaler = StandardScaler()
    scaler.mean_ = app.config['MEAN']
    scaler.var_ = app.config['VAR']
    scaler.scale_ = app.config['SCALE']
    img = scaler.transform(img.reshape(1, -1))
    # print(img)
    res = {
        "x":myModel.classes_.tolist(),
        "y":myModel.predict_proba(img)[0].tolist(),
        "type": "bar"
    }
    # print(res)
    return res


def load_model_from_file():
    myModel = load('logres.m5')
    myModelName = 'Logistic Regression'
    myMean = np.load('logres_mean.npy')
    myVar = np.load('logres_var.npy')
    myScale = np.load('logres_scale.npy')
    return (myModel, myModelName, myMean, myVar, myScale)

def init():
    (myModel, myModelName, myMean, myVar, myScale) = load_model_from_file()
    app.config['MODEL'] = myModel
    app.config['MODELNAME'] = myModelName
    app.config['MEAN'] = myMean
    app.config['VAR'] = myVar
    app.config['SCALE'] = myScale

# app.run()