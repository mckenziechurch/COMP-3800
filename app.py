import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
CLFmodel = pickle.load(open('model.pkl', 'rb'))  # load the ML model

# The route() decorator to tell Flask what URL should trigger our function.
# ‘/’ is the root of the website, such as www.westga.edu
@flask_app.route("/")  

# function to read the html file
def index():
    return render_template("index.html")

# at thie endpoint, do the POST request from the Flask API
@flask_app.route("/predict", methods = ["POST"])   

# method to make prediction using the pickle file model from user input in index.html
def predict():
     # read features from the input form in index
    float_features = [int(x) for x in request.form.values()]

    # create array from the input
    features = [np.array(float_features)]

    # pass the input array to the prediction model to make prediction
    result = CLFmodel.predict(features)

    # return prediction result to index file
    return render_template("index.html", predicted_text = result) # load the prediction in the front-end (index.html)

if __name__ =="__main__":
    flask_app.run(debug = True)
