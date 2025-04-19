from flask import Flask

app = Flask(__name__)

@app.route("/")

def hello_word():
    f = open('index.html', 'r')
    return f
