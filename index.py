from flask import Flask
import os
from flask import render_template, flash, request, redirect, url_for
from joblib import dump, load


app = Flask(__name__, template_folder='')

@app.route("/")
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    else: #if method == 'POST'
        img = request.image
        return redirect(url_for('predict', imagedata=img))


def load_model_from_file():
    myModel = load('logres.m5')
    return myModel

def predict(imgdata):
    myModel = app.config['MODEL']
    res = myModel.predict(imgdata)
    return render_template('index.html',result=res)

def main():
    myModel = load_model_from_file()
    app.config['MODEL'] = myModel
    app.run()

main()