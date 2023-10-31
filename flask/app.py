import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)

CORS(app)

with open('./model/saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)
    regressor_loaded = data["model"]
    le_country = data["le_country"]
    le_education = data["le_education"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    country = request.form.get('country')
    ed_level = request.form.get('ed_level')
    years_code = request.form.get('years_code')

    x = np.array([[country, ed_level, years_code]])
    x[:, 0] = le_country.transform(x[:, 0])
    x[:, 1] = le_education.transform(x[:, 1])
    x = x.astype(float)

    y_pred = regressor_loaded.predict(x)

    return render_template('predict.html', data=y_pred)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html'), 404
 
@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html'), 500

if __name__ == "__main__":
    app.run()