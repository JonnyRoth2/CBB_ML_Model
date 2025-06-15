from flask import Flask, request, render_template, send_from_directory
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from predictor import predict_winner

app = Flask(__name__, static_folder='public', static_url_path='')

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')
@app.route('/predict', methods=['POST'])
def predict():
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')

    if not team1 or not team2:
        return "Both teams must be selected.", 400
    prediction = predict_matchup(team1, team2)

    return f"The predicted winner is {prediction['winner']} with {prediction['confidence']}% confidence."

def predict_matchup(home_team, away_team):

    winner,confidence=predict_winner(home_team, away_team,'n')
    return {'winner': winner, 'confidence': confidence}
if __name__ == '__main__':
    app.run(debug=True)