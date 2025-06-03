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


def create_matchup_features(df):
    team_stats = {}
    for team in df['Team'].unique():
        team_data = df[df['Team'] == team]
        team_stats[team] = {
            'Avg_Adj_O': team_data['Adj. O'].mean(),
            'Avg_Adj_D': team_data['Adj. D'].mean(),
            'Avg_T': team_data['T'].mean(),
            'Avg_eFG_O': team_data['eFG%_O'].mean(),
            'Avg_TO_O': team_data['TO%_O'].mean(),
            'Avg_Reb_O': team_data['Reb%_O'].mean(),
            'Avg_FTR_O': team_data['FTR_O'].mean(),
            'Avg_eFG_D': team_data['eFG%_D'].mean(),
            'Avg_TO_D': team_data['TO%_D'].mean(),
            'Avg_Reb_D': team_data['Reb%_D'].mean(),
            'Avg_FTR_D': team_data['FTR_D'].mean(),
        }
    matchup_df = df.copy()
    opp_columns = [
        'Opp_Avg_Adj_O', 'Opp_Avg_Adj_D', 'Opp_Avg_T',
        'Opp_Avg_eFG_O', 'Opp_Avg_TO_O', 'Opp_Avg_Reb_O', 'Opp_Avg_FTR_O',
        'Opp_Avg_eFG_D', 'Opp_Avg_TO_D', 'Opp_Avg_Reb_D', 'Opp_Avg_FTR_D'
    ]

    team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index').reset_index()
    team_stats_df.rename(columns={'index': 'Opp.'}, inplace=True)
    team_stats_df = team_stats_df.rename(columns=lambda x: f"Opp_{x}" if x != 'Opp.' else x)
    matchup_df = matchup_df.merge(team_stats_df, how='left', on='Opp.')
    # display(matchup_df)
    # display(team_stats_df)
    matchup_df.fillna(0, inplace=True)
    matchup_df['Offense_vs_Defense'] = matchup_df['Adj. O'] - matchup_df['Opp_Avg_Adj_D']
    matchup_df['Defense_vs_Offense'] = matchup_df['Adj. D'] - matchup_df['Opp_Avg_Adj_O']
    matchup_df['Tempo_Differential'] = matchup_df['T'] - matchup_df['Opp_Avg_T']
    matchup_df['eFG_Advantage_O'] = matchup_df['eFG%_O'] - matchup_df['Opp_Avg_eFG_D']
    matchup_df['TO_Advantage_O'] = matchup_df['Opp_Avg_TO_D'] - matchup_df['TO%_O']
    matchup_df['Reb_Advantage_O'] = matchup_df['Reb%_O'] - matchup_df['Opp_Avg_Reb_D']
    matchup_df['FTR_Advantage_O'] = matchup_df['FTR_O'] - matchup_df['Opp_Avg_FTR_D']

    matchup_df['eFG_Advantage_D'] = matchup_df['Opp_Avg_eFG_O'] - matchup_df['eFG%_D']
    matchup_df['TO_Advantage_D'] = matchup_df['TO%_D'] - matchup_df['Opp_Avg_TO_O']
    matchup_df['Reb_Advantage_D'] = matchup_df['Opp_Avg_Reb_O'] - matchup_df['Reb%_D']
    matchup_df['FTR_Advantage_D'] = matchup_df['Opp_Avg_FTR_O'] - matchup_df['FTR_D']
    matchup_df['True Advantage'] = matchup_df['True Advantage'].astype(float)
    return matchup_df

def process_data():
    historical_df=pd.read_csv('beginningHist.csv')
    current_df=pd.read_csv('beginningCurr.csv')
    temp_hc_ad_df=pd.read_csv('hca.csv')
    historical_df.columns = historical_df.iloc[0]
    historical_df.drop(historical_df.head(1).index,inplace=True)
    historical_df.drop(historical_df.tail(1).index,inplace=True)
    historical_df.drop(historical_df.columns[-1],inplace=True,axis=1)
    historical_df.drop(historical_df.columns[0],inplace=True, axis=1)
    current_df.columns = current_df.iloc[0]
    current_df.drop(current_df.head(1).index,inplace=True)
    current_df.drop(current_df.tail(1).index,inplace=True)
    current_df.drop(current_df.columns[-1],inplace=True,axis=1)
    current_df.drop(current_df.columns[0],inplace=True, axis=1)
    

    name_changes={
        "NC State": "N.C. State",
        "Indiana St": "Indiana St.",
        "San Diego St": "San Diego St.",
        "Mississippi St": "Mississippi St.",
        "North Carolina St": "North Carolina St.",
        "Penn St": "Penn St.",
        "Ohio St": "Ohio St.",
        "Ohio State": "Ohio St.",
        "Iowa St": "Iowa St.",
        "Boise St": "Boise St.",
        "Fresno St": "Fresno St.",
        "Arizona St": "Arizona St.",
        "Texas AM": "Texas A&M",
        "Utah St": "Utah St.",
        "Texas St": "Texas St.",
        "Michigan St": "Michigan St.",
        "Washington St": "Washington St.",
        "Kansas St": "Kansas St.",
        "Miami Florida": "Miami FL",
        "Texas Christian": "TCU",
        "Northwestern St": "Northwestern St.",
        "McNeese St": "McNeese St.",
        "Morehead St": "Morehead St.",
        "Wisconsin St": "Wisconsin St.",
        "South Dakota State": "South Dakota St.",
        "Colorado St": "Colorado St.",
        "Long Beach St": "Long Beach St.",
        "Brigham Young": "BYU",
        "Florida St": "Florida St.",
        "Louisiana State": "LSU",
        "Southern California": "USC",
        "Central Florida": "UCF",
        "Louisiana Tech": "UT",
        "Missouri St": "Missouri St.",
        "Alabama St": "Alabama St.",
        "Oklahoma St": "Oklahoma St.",
        "Citadel": "The Citadel",
        "Arkansas St": "Arkansas St.",
        "Murray St": "Murray St.",
        "Southern Methodist": "SMU",
        "St Johns": "St. John's",
        "San Jose St": "San Jose St.",
        "Pennsylvania": "Penn",
        "St Bonaventure": "St. Bonaventure",
        "Oregon St": "Oregon St.",
        "Wichita St": "Wichita St.",
    }

    temp_hc_ad_df['Team']=temp_hc_ad_df['Team'].replace(name_changes)
    historical_df = historical_df.merge(temp_hc_ad_df[['Team','True Advantage']], on='Team', how='left')
    nan_rows = historical_df[historical_df['True Advantage'].isna()]
    historical_df.dropna(subset=['True Advantage'],inplace=True)
    #temp_hc_ad_df['Team']=temp_hc_ad_df['Team'].replace(name_changes)
    current_df = current_df.merge(temp_hc_ad_df[['Team','True Advantage']], on='Team', how='left')
    nan_rows = current_df[current_df['True Advantage'].isna()]
    current_df.dropna(subset=['True Advantage'],inplace=True)
    historical_df["hca"]=historical_df["True Advantage"]
    current_df["hca"]=current_df["True Advantage"]
    historical_df.columns=['Rk', 'Date', 'Type', 'Team', 'Conf.', 'Opp.', 'Venue', 'Result',
       'Adj. O', 'Adj. D', 'T', 'EFF_O', 'eFG%_O', 'TO%_O', 'Reb%_O', 'FTR_O', 'EFF_D',
       'eFG%_D', 'TO%_D', 'Reb%_D', 'FTR_D', 'G-Sc', '+/-', 'True Advantage','hca']

    historical_df.loc[historical_df['Venue'].str.upper() != 'H', 'True Advantage'] = 0
    current_df.columns=['Rk', 'Date', 'Type', 'Team', 'Conf.', 'Opp.', 'Venue', 'Result',
        'Adj. O', 'Adj. D', 'T', 'EFF_O', 'eFG%_O', 'TO%_O', 'Reb%_O', 'FTR_O', 'EFF_D',
        'eFG%_D', 'TO%_D', 'Reb%_D', 'FTR_D', 'G-Sc', '+/-', 'True Advantage','hca']

    current_df.loc[current_df['Venue'].str.upper() != 'H', 'True Advantage'] = 0
    historical_df['Won/Loss']=historical_df['Result'].str[0]
    temp=historical_df["Result"].str.split(', ', expand=True)
    historical_df["Score Range"]=temp[1]
    temp2=historical_df["Score Range"].str.split('-', expand=True).astype(int)

    def score_fun(row):
        (s1,s2)=row['Score Range'].split('-')
        (s1,s2)=int(s1),int(s2)
        if row['Won/Loss']=='W':
            return max(s1,s2)
        else:
            return min(s1,s2)
    def opp_score_fun(row):
        (s1,s2)=row['Score Range'].split('-')
        (s1,s2)=int(s1),int(s2)
        if row['Won/Loss']=='W':
            return min(s1,s2)
        else:
            return max(s1,s2)
    historical_df['Score']=historical_df.apply(score_fun,axis=1)
    historical_df['Opp_Score']=historical_df.apply(opp_score_fun,axis=1)

    current_df['Won/Loss']=current_df['Result'].str[0]
    temp=current_df["Result"].str.split(', ', expand=True)
    current_df["Score Range"]=temp[1]
    temp2=current_df["Score Range"].str.split('-', expand=True).astype(int)


    current_df['Score']=current_df.apply(score_fun,axis=1)
    historical_df['Opp_Score']=historical_df.apply(opp_score_fun,axis=1)
    current_df['Opp_Score']=current_df.apply(opp_score_fun,axis=1)
    current_df['Spread']=current_df['Score']-current_df['Opp_Score']
    historical_df['Spread']=historical_df['Score']-historical_df['Opp_Score']
 
    historical_df.to_csv("Cleaned_Historicals.csv",index=False)
    current_df.to_csv("Cleaned_2025.csv",index=False)
