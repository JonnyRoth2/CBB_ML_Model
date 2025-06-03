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
from data_proc import create_matchup_features

def train_logistic_regression(X_train_scaled,X_test_scaled,y_train_winner,y_test_winner):
    print("\n=== Training Logistic Regression Model for Win Prediction ===")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train_winner)

    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train_winner, train_preds)
    test_accuracy = accuracy_score(y_test_winner, test_preds)

    train_report = classification_report(y_train_winner, train_preds)
    test_report = classification_report(y_test_winner, test_preds)
    print(f"Training accuracy: {train_accuracy}")
    print(f"Testing accuracy: {test_accuracy}")
    print(f"Classification Report: {train_report}")
    print(f"Classification Report: {test_report}")
    cm = confusion_matrix(y_test_winner, test_preds)
    return model, train_accuracy, test_accuracy, cm

def train_random_forest(X_train_scaled,X_test_scaled,y_train_spread,y_train_winner,y_test_winner,y_test_spread):
    print("\n=== Training Random Forest Models ===")

    model = RandomForestClassifier(n_estimators=1000, max_depth=8)
    model.fit(X_train_scaled, y_train_winner)

    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=8,min_samples_leaf=5)
    rf_regressor.fit(X_train_scaled, y_train_spread)

    train_win_preds = model.predict(X_train_scaled)
    test_win_preds = model.predict(X_test_scaled)

    train_spread_preds = rf_regressor.predict(X_train_scaled)
    test_spread_preds = rf_regressor.predict(X_test_scaled)

    train_win_accuracy = accuracy_score(y_train_winner, train_win_preds)
    test_win_accuracy = accuracy_score(y_test_winner, test_win_preds)

    train_report = classification_report(y_train_winner, train_win_preds)
    test_report = classification_report(y_test_winner, test_win_preds)
    print(f"Random Forest Win Prediction:")
    print(f"Training accuracy: {train_win_accuracy}")
    print(f"Testing accuracy: {test_win_accuracy}")
    print(f"Classification Report: {train_report}")
    print(f"Classification Report: {test_report}")
    train_spread_mae = mean_absolute_error(y_train_spread, train_spread_preds)
    test_spread_mae = mean_absolute_error(y_test_spread, test_spread_preds)
    train_spread_r2 = r2_score(y_train_spread, train_spread_preds)
    test_spread_r2 = r2_score(y_test_spread, test_spread_preds)
    print(f"Random Forest Spread Prediction:")
    print(f"Training MAE: {train_spread_mae}")
    print(f"Testing MAE: {test_spread_mae:}")
    print(f"Training R2: {train_spread_r2}")
    print(f"Testing R2: {test_spread_r2:}")
    cm = confusion_matrix(y_test_winner, test_win_preds)
    return model, rf_regressor, train_win_accuracy, test_win_accuracy, train_spread_mae, test_spread_mae, cm

def train_grad_boost(X_train_scaled,X_test_scaled,y_train_spread,y_train_winner,y_test_winner,y_test_spread):
    print("\n=== Training Gradient Boost Models ===")

    model = GradientBoostingClassifier()
    model.fit(X_train_scaled, y_train_winner)

    gb_regressor = GradientBoostingRegressor()
    gb_regressor.fit(X_train_scaled, y_train_spread)

    train_win_preds = model.predict(X_train_scaled)
    test_win_preds = model.predict(X_test_scaled)

    train_spread_preds = gb_regressor.predict(X_train_scaled)
    test_spread_preds = gb_regressor.predict(X_test_scaled)

    train_win_accuracy = accuracy_score(y_train_winner, train_win_preds)
    test_win_accuracy = accuracy_score(y_test_winner, test_win_preds)

    train_report = classification_report(y_train_winner, train_win_preds)
    test_report = classification_report(y_test_winner, test_win_preds)
    print(f"Gradient Boost Win Prediction:")
    print(f"Training accuracy: {train_win_accuracy}")
    print(f"Testing accuracy: {test_win_accuracy}")
    print(f"Classification Report: {train_report}")
    print(f"Classification Report: {test_report}")
    train_spread_mae = mean_absolute_error(y_train_spread, train_spread_preds)
    test_spread_mae = mean_absolute_error(y_test_spread, test_spread_preds)
    train_spread_r2 = r2_score(y_train_spread, train_spread_preds)
    test_spread_r2 = r2_score(y_test_spread, test_spread_preds)
    print(f"Random Forest Spread Prediction:")
    print(f"Training MAE: {train_spread_mae}")
    print(f"Testing MAE: {test_spread_mae:}")
    print(f"Training R2: {train_spread_r2}")
    print(f"Testing R2: {test_spread_r2:}")
    cm = confusion_matrix(y_test_winner, test_win_preds)
    return model, gb_regressor, train_win_accuracy, test_win_accuracy,train_spread_mae, test_spread_mae, cm, train_spread_r2, test_spread_r2

def train_models():
    historical_data = pd.read_csv('Cleaned_Historicals.csv')
    current_data = pd.read_csv('Cleaned_2025.csv')
    numeric_cols = [
        'Adj. O', 'Adj. D', 'T',
        'eFG%_O', 'TO%_O', 'Reb%_O', 'FTR_O',
        'eFG%_D', 'TO%_D', 'Reb%_D', 'FTR_D', 'hca'
    ]
    for col in numeric_cols:
        historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')
        current_data[col] = pd.to_numeric(current_data[col], errors='coerce')
    historical_matchups = create_matchup_features(historical_data)
    current_matchups = create_matchup_features(current_data)
    all_features =  [
        'Adj. O', 'Adj. D', 'T',
        'eFG%_O', 'TO%_O', 'Reb%_O', 'FTR_O',
        'eFG%_D', 'TO%_D', 'Reb%_D', 'FTR_D',
        'hca',  'Offense_vs_Defense', 'Defense_vs_Offense', 'Tempo_Differential',
        'eFG_Advantage_O', 'TO_Advantage_O', 'Reb_Advantage_O', 'FTR_Advantage_O',
        'eFG_Advantage_D', 'TO_Advantage_D', 'Reb_Advantage_D', 'FTR_Advantage_D','True Advantage'
    ]
    # all_features.remove('Won/Loss')
    X_train = historical_matchups[all_features].copy()
    # print('xtrain')
    # display(X_train)
    y_train_winner = (historical_matchups['Won/Loss'] == 'W').astype(int)
    y_train_spread = historical_matchups['Spread']
    y_train_team_score = historical_matchups['Score']
    y_train_opp_score = historical_matchups['Opp_Score']
    X_test = current_matchups[all_features].copy()
    y_test_winner = (current_matchups['Won/Loss'] == 'W').astype(int)
    y_test_spread = current_matchups['Spread']
    y_test_team_score = current_matchups['Score']
    y_test_opp_score = current_matchups['Opp_Score']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_result = train_logistic_regression(X_train_scaled,X_test_scaled,y_train_winner,y_test_winner)

    rf_results = train_random_forest(X_train_scaled,X_test_scaled,y_train_spread,y_train_winner,y_test_winner,y_test_spread)
    gb_results = train_grad_boost(X_train_scaled,X_test_scaled,y_train_spread,y_train_winner,y_test_winner,y_test_spread)
    joblib.dump(log_result[0],'log_model.pkl')
    joblib.dump(rf_results[0],'rf_model.pkl')
    joblib.dump(gb_results[0],'gb_model.pkl')
    joblib.dump(gb_results[1],'gb_reg.pkl')
    joblib.dump(scaler, 'scaler.pkl')
train_models()