from flask import Flask
import os
from flask import render_template, flash, request, redirect, url_for
from joblib import dump, load


app = Flask(__name__, template_folder='')

@app.route("/")
def hello_world():
    return render_template('index.html')


def load_model_from_file():
    myModel = load('logres.m5')
    return myModel

def main():
    myModel = load_model_from_file()
    app.config['MODEL'] = myModel
    app.run()

main()