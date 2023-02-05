import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)

## loading the model
regmodel = pickle.load(open('regression.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    print('executing')
    data = request.json["data"]
    print(data)

    print(np.array(list(data.values())).reshape(1, -1))

    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))

    output = regmodel.predict(new_data)
    output = "{} percentage".format(round(list(output[0])[0], 2))

    print(output)
    return jsonify(output)

@app.route('/predict', methods = ['POST'])
def predict():
    print('executing')
    for i in request.form.values():
        print(i)
    data = [float(x) for x in request.form.values()]
    print(data)
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = list(regmodel.predict(final_input)[0])
    print(output[0])
    return render_template('home.html', prediction_text = 'The body fat is {} %'.format(round(output[0], 2)))


if __name__ == "__main__":
    app.run(debug=True)
