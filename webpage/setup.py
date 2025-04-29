from flask import Flask


app = Flask(__name__)

@app.route("/")
def index():
    with open('index.html', 'r') as f:
        return f.read()


@app.route("/style.css")
def style():
    with open('style.css', 'r') as f:
        return f.read()


@app.route("/script.js")
def script():
    with open('script.js', 'r') as f:
        return f.read()

