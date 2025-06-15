import time
import io
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.inspection import permutation_importance
import joblib
import os


from data_proc import process_data

# process_data()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'Cleaned_2025.csv')
current_data = pd.read_csv(csv_path)
numeric_cols = [
    'Adj. O', 'Adj. D', 'T',
    'eFG%_O', 'TO%_O', 'Reb%_O', 'FTR_O',
    'eFG%_D', 'TO%_D', 'Reb%_D', 'FTR_D', 'hca'
]
log_model_path=os.path.join(BASE_DIR, 'log_model.pkl')
log_model=joblib.load(log_model_path)
gb_model_path=os.path.join(BASE_DIR, 'gb_reg.pkl')
gb_reg=joblib.load(gb_model_path)
scaler_path=os.path.join(BASE_DIR, 'scaler.pkl')
scaler=joblib.load(scaler_path)


def two_team_matchup(team1, team2):
    team1=team1.upper()
    team2=team2.upper()
    current_data['Team'] = current_data['Team'].str.upper()
    team1_data = current_data[current_data['Team'] == team1]
    team2_data = current_data[current_data['Team'] == team2]
    if team1_data.empty or team2_data.empty:
        print("One or both teams not found in the dataset.")
        return
    team1_avg = team1_data[numeric_cols].mean()
    team2_avg = team2_data[numeric_cols].mean()
    #display(team1_avg)
    #display(team2_avg)
    features = {
        'Adj. O': team1_avg['Adj. O'],
        'Adj. D': team1_avg['Adj. D'],
        'T': team1_avg['T'],
        'eFG%_O': team1_avg['eFG%_O'],
        'TO%_O': team1_avg['TO%_O'],
        'Reb%_O': team1_avg['Reb%_O'],
        'FTR_O': team1_avg['FTR_O'],
        'eFG%_D': team1_avg['eFG%_D'],
        'TO%_D': team1_avg['TO%_D'],
        'Reb%_D': team1_avg['Reb%_D'],
        'FTR_D': team1_avg['FTR_D'],
        'hca': team1_avg['hca'],
        'Offense_vs_Defense': team1_avg['Adj. O'] - team2_avg['Adj. D'],
        'Defense_vs_Offense': team1_avg['Adj. D'] - team2_avg['Adj. O'],
        'Tempo_Differential': team1_avg['T'] - team2_avg['T'],
        'eFG_Advantage_O': team1_avg['eFG%_O'] - team2_avg['eFG%_D'],
        'TO_Advantage_O': team2_avg['TO%_D'] - team1_avg['TO%_O'],
        'Reb_Advantage_O': team1_avg['Reb%_O'] - team2_avg['Reb%_D'],
        'FTR_Advantage_O': team1_avg['FTR_O'] - team2_avg['FTR_D'],
        'eFG_Advantage_D': team2_avg['eFG%_O'] - team1_avg['eFG%_D'],
        'TO_Advantage_D': team1_avg['TO%_D'] - team2_avg['TO%_O'],
        'Reb_Advantage_D': team2_avg['Reb%_O'] - team1_avg['Reb%_D'],
        'FTR_Advantage_D': team2_avg['FTR_O'] - team1_avg['FTR_D'],
        'True Advantage': 0
    }
    matchup_df = pd.DataFrame([features])
    matchup_df.fillna(0, inplace=True)

    return matchup_df

def predict_winner(team1, team2, home):
    team1=team1.upper()
    team2=team2.upper()
    current_data['Team'] = current_data['Team'].str.upper()
    team1_data = current_data[current_data['Team'] == team1]
    team2_data = current_data[current_data['Team'] == team2]
    if team1_data.empty or team2_data.empty:
        print("One or both teams not found in the dataset.")
        return
    matchup1 = two_team_matchup(team1, team2)
    matchup2 = two_team_matchup(team2, team1)
    if home.lower() == 'h':
      matchup1['True Advantage'] = team1_data['hca'].mean()
      matchup2['True Advantage'] = 0
    elif home.lower() == 'a':
      matchup1['True Advantage'] = 0
      matchup2['True Advantage'] = team2_data['hca'].mean()
    else:
      matchup1['True Advantage'] = 0
      matchup2['True Advantage'] = 0
    # team1_avg = team1_matchup[all_features].mean()
    # team2_avg = team2_matchup[all_features].mean()
    # X = matchup
    matchup1.fillna(0, inplace=True)
    matchup2.fillna(0, inplace=True)
    m1_scaled = scaler.transform(matchup1)
    m2_scaled = scaler.transform(matchup2)
    prediction1 = log_model.predict_proba(m1_scaled)
    prediction2 = log_model.predict_proba(m2_scaled)
    team1_win_prob = prediction1[0][1]
    team2_win_prob = prediction2[0][1]
    # print(f"Logistic Regression Win Probability for {team1}: {team1_win_prob}")
    # print(f"Logistic Regression Win Probability for {team2}: {team2_win_prob}")
    if team1_win_prob>team2_win_prob:
      print(f"Most likely for {team1} to win in Log Reg")
      return [team1, team1_win_prob]
    else:
      print(f"Most likely for {team2} to win in Log Reg")
      return [team2, team2_win_prob]
    # matchup1.to_csv('matchup.csv', index=False)
    # display(team1_matchup)
    # display(team2_matchup)
    

def predict_spread(team1, team2, home):

    team1=team1.upper()
    team2=team2.upper()
    current_data['Team'] = current_data['Team'].str.upper()
    team1_data = current_data[current_data['Team'] == team1]
    team2_data = current_data[current_data['Team'] == team2]
    if team1_data.empty or team2_data.empty:
        print("One or both teams not found in the dataset.")
        return
    matchup1 = two_team_matchup(team1, team2)
    matchup2 = two_team_matchup(team2, team1)
    if home.lower() == 'h':
      matchup1['True Advantage'] = team1_data['hca'].mean()
      matchup2['True Advantage'] = 0
    elif home.lower() == 'a':
      matchup1['True Advantage'] = 0
      matchup2['True Advantage'] = team2_data['hca'].mean()
    else:
      matchup1['True Advantage'] = 0
      matchup2['True Advantage'] = 0

    matchup1.fillna(0, inplace=True)
    matchup2.fillna(0, inplace=True)
    m1_scaled = scaler.transform(matchup1)
    m2_scaled = scaler.transform(matchup2)
    prediction1 = gb_reg.predict(m1_scaled)
    prediction2 = gb_reg.predict(m2_scaled)
    spread = prediction1[0] - prediction2[0]

    if spread > 0:
      print(f"Most likely for {team1} to win in Gradient Boost by {spread} points")
      return team1,-spread
    else:
      print(f"Most likely for {team2} to win in Gradient Boost by {-spread} points")
      return team2,-spread

# t1 = input("What is team 1? ")
# t2 = input("What is team 2? ")
# home = input("Is team 1 home? (h for home a for away, n for neutral) ")
# predict_winner(t1,t2,home)
# predict_spread(t1,t2,home)