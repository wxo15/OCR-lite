from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return '<h1>OCR-lite</h1><canvas width="280" height="280"></canvas>'

app.run()
