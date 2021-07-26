from flask import Flask
import os
from io import BytesIO
from flask import render_template, flash, request, redirect, url_for
from joblib import dump, load
import numpy as np
import boto3
from sklearn.preprocessing import StandardScaler

if os.environ.get('IS_HEROKU', None):
    s3 = boto3.resource('s3',
        aws_access_key_id= process.env.AWS_ACCESS_KEY_ID,
        aws_secret_access_key= process.env.AWS_SECRET_ACCESS_KEY)
else:
    from keys import awsaccesskey, awssecretkey

    s3 = boto3.resource('s3',
        aws_access_key_id= awsaccesskey(),
        aws_secret_access_key=awssecretkey())


app = Flask(__name__, template_folder='')

@app.route("/", methods=['GET']) 
def render():
    init()
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
    with BytesIO() as data:
        s3.Bucket('ocr-lite').download_fileobj('logres.m5', data)
        data.seek(0)
        myModel = load(data)
    myModelName = 'Logistic Regression'
    myMean = np.frombuffer(s3.Object('ocr-lite', 'logres_mean.npy').get()['Body'].read())[16:]
    myVar = np.frombuffer(s3.Object('ocr-lite', 'logres_var.npy').get()['Body'].read())[16:]
    myScale = np.frombuffer(s3.Object('ocr-lite', 'logres_scale.npy').get()['Body'].read())[16:]
    # myModel = load('logres.m5')
    # myMean = np.load('logres_mean.npy')
    # myVar = np.load('logres_var.npy')
    # myScale = np.load('logres_scale.npy')
    return (myModel, myModelName, myMean, myVar, myScale)

def init():
    (myModel, myModelName, myMean, myVar, myScale) = load_model_from_file()
    app.config['MODEL'] = myModel
    app.config['MODELNAME'] = myModelName
    app.config['MEAN'] = myMean
    app.config['VAR'] = myVar
    app.config['SCALE'] = myScale

if not(os.environ.get('IS_HEROKU', None)):
    app.run()
