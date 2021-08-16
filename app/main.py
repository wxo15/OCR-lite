from flask import Flask
import os
from io import BytesIO
from flask import render_template, flash, request, redirect, url_for
from joblib import dump, load
import numpy as np
import boto3
import s3fs
import zipfile
import tempfile
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

if os.environ.get('IS_HEROKU', None):
    s3 = boto3.resource('s3',
        aws_access_key_id= os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key= os.environ.get('AWS_SECRET_ACCESS_KEY'))
else:
    from keys import awsaccesskey, awssecretkey

    s3 = boto3.resource('s3',
        aws_access_key_id= awsaccesskey(),
        aws_secret_access_key=awssecretkey())

    s3fs1 = s3fs.S3FileSystem(key=awsaccesskey(), secret=awssecretkey())

app = Flask(__name__, template_folder='')

@app.route("/", methods=['GET']) 
def render():
    if not('MODELNAME' in app.config.keys()):
        init_CNN()
    elif app.config['MODELNAME'] == 'Convolutional Neural Network':
        init_logres()
    else:
        init_CNN()
    return render_template('index.html', modelName=app.config['MODELNAME'])

@app.route("/predict", methods=['POST']) 
def predict():
    if request.json['image'] is None:
            flash('No file part')
            return redirect(request.url)
    img = np.array(request.json['image'])
    modelName = app.config['MODELNAME']
    if modelName == 'Logistic Regression':
        # print(request.json['image'])
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
    elif modelName == 'Convolutional Neural Network':
        myModel = app.config['MODEL']
        res = {
            "x":list(range(10)),
            "y":myModel.predict((img/255).reshape(1,28,28,1)).flatten().tolist(),
            "type": "bar"
        }
        return res


def load_logres_from_file():
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

def init_logres():
    (myModel, myModelName, myMean, myVar, myScale) = load_logres_from_file()
    app.config['MODEL'] = myModel
    app.config['MODELNAME'] = myModelName
    app.config['MEAN'] = myMean
    app.config['VAR'] = myVar
    app.config['SCALE'] = myScale

def load_CNN_from_file():
    with tempfile.TemporaryDirectory() as tempdir:
        result = s3.download_file("ocr-lite",'convolutional.h5', tempdir + "/convolutional.h5")
        myModel = load_model(tempdir + "/convolutional.h5")
        # myModel = load_model('convolutional.h5')
        return myModel

def s3_get_keras_model():
    with tempfile.TemporaryDirectory() as tempdir:
        # Fetch and save the zip file to the temporary directory
        s3fs1.get("s3://ocr-lite/convolutional.h5", tempdir+"/convolutional.h3")
        # Load the keras model from the temporary directory
        return load_model(tempdir+"/convolutional.h3")


def init_CNN():
    myModel = s3_get_keras_model()
    app.config['MODEL'] = myModel
    app.config['MODELNAME'] = 'Convolutional Neural Network'
    app.config['MEAN'] = None
    app.config['VAR'] = None
    app.config['SCALE'] = None

if not(os.environ.get('IS_HEROKU', None)):
    app.run()


