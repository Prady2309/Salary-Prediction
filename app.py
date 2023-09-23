import numpy as np
# import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)   # Constructor
model = pickle.load(open('salary.pkl', 'rb'))

# What to display when nothing is specified
@app.route('/')
def home():
    return render_template('index.html')


#Prediction function
def ValPredictor(to_predict_list) :       # User response
    # creating a numpy array of the USER RESPONSES and reshaping
    to_predict = np.array(to_predict_list).reshape(1,12) 
    result = model.predict(to_predict)     # Predicting
    return result[0]       # returning only the value

# This function will be called after submitting the form.!!!
@app.route('/predict', methods = ['POST'])
def predict() :
    if request.method == 'POST' :
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        result = ValPredictor(final_features)
        # we are using logistic Regression for prediction
        # the output = 0/1
        if int(result) == 1 :
            prediction = "Income more than 50K"
        else :
            prediction = "Income less than 50K"
        #Result page will show the prediction
        return render_template("result.html", prediction = prediction)

@app.route('/result',methods=['POST'])
def result():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

# MAIN FUNCTION
if __name__ == '__main__' :
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] == True
            
