from flask import Flask, request, jsonify
from model import predictor

app = Flask(__name__)
@app.route('/predict',methods=['POST'])

def predict():
    data = request.get_json()
    home_team = data['home_team']
    away_team = data['away_team']

    prediction = predictor.predict_winner(home_team,away_team,'n')

    return jsonify({
        'predicted_winner': prediction[0],
        'confidence': prediction[1]
    })