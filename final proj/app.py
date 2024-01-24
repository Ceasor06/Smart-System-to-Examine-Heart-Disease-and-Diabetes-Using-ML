from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
# Load the saved model
#model = joblib.load('logistic_regression_model.joblib', 'rb')
model = pickle.load(open('app142.pkl', 'rb'))
# Define a Flask app
app = Flask(__name__)

# Define the home page route
@app.route('/', methods=['GET'])
# @app.route('/')
def index():
    return render_template('indx.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def Predict():
    # Get the input data as a dictionary
    #data = request.form.to_dict()
    d1 = int(request.form.get('age'))
    d2 = int(request.form.get('sex'))
    d3 = int(request.form.get('cp'))
    d4 = int(request.form.get('trestbps'))
    d5 = int(request.form.get('chol'))
    d6 = int(request.form.get('fbs'))
    d7 = int(request.form.get('restecg'))
    d8 = int(request.form.get('thalach'))
    d9 = int(request.form.get('exang'))
    d10 = float(request.form.get('oldpeak'))
    d11 = int(request.form.get('slope'))
    d12 = int(request.form.get('ca'))
    d13 = int(request.form.get('thal'))
    # Prediction
    result=model.predict(np.array([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13]).reshape(1,13))
    return str(result)
    # Convert the data into a numpy array
    # data_array = np.array(list(data.values())).reshape(1, -1).astype(float)
    # Use the loaded model to make a prediction
    # prediction = loaded_model.predict(data_array)
    # Return the prediction as a JSON object
    # return jsonify(int(prediction[0]))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5050)
    #app.run(debug=True, port=8080)
