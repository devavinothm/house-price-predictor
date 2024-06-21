import pandas as pd
import pickle

from flask import Flask, request, render_template

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
model = pickle.load(open('ridge_model.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    total_sqft = request.form.get('total_sqft')
    bath = request.form.get('bath')
    bhk = request.form.get('bhk')

    input = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    prediction = model.predict(input)

    prediction_value = prediction[0] * 100000

    formated_prediction = "{:,.2f}".format(prediction_value)

    print(prediction[0])

    return render_template('index.html', prediction=formated_prediction)

if __name__ == '__main__':
    app.run(debug=True)