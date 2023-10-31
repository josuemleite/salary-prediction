import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)

CORS(app)

model = pickle.load(open("./model/saved_steps.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()