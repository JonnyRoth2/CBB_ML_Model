from flask import Flask, request, render_template, send_from_directory


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

    import random
    confidence = round(random.uniform(50, 100), 2)
    winner = home_team if random.random() > 0.5 else away_team
    return {'winner': winner, 'confidence': confidence}
if __name__ == '__main__':
    app.run(debug=True)