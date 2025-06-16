from flask import Flask, request, render_template, send_from_directory
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from predictor import predict_winner, predict_spread

app = Flask(__name__, static_folder='public', static_url_path='')

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')
@app.route('/predictor')
def predictor():
    return render_template('predictor.html')
@app.route('/predict', methods=['POST'])
def predict():
    team1 = request.form.get('team1')
    team2 = request.form.get('team2')
    hca = request.form.get('hca')
    if not team1 or not team2:
        return "Both teams must be selected.", 400

    winner, confidence = predict_winner(team1, team2, hca)
    gb_winner, spread = predict_spread(team1, team2, hca)
    return f"The predicted winner is {winner} with {confidence}% confidence. \n The predicted spread and winner through regression model is {gb_winner} with a spread of {spread}."

if __name__ == '__main__':
    app.run(debug=True)