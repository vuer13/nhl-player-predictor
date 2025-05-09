import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def predict():
    df_2024 = processData()
    model = joblib.load('../models/xgb_model_goals.pkl')

    features = ["age", "age2", 'pos_C', 'pos_D', 'pos_L', 'pos_R', 'games_played', 'xGoals',
            'pAssists', 'sAssists', 'sog', 'points', 'goals', 'on_ice_chances', 'on_ice_goals', 
            'icetime_per_game', 'shot_percentage', 'points_per_60', 'ixG-goals', 'ppg', 'apg', 'gpg',
            'points_lag_1', 'points_lag_2', 'points_lag_3', 'points_lag_4', 'points_lag_5', 
            'ppg_lag_1', 'ppg_lag_2', 'ppg_lag_3', 'ppg_lag_4', 'ppg_lag_5', 'pAssists_lag_1', 'sAssists_lag_1',
            'goals_lag_1', 'goals_lag_2', 'goals_lag_3', 'goals_lag_4', 'goals_lag_5', 
            'gpg_lag_1', 'gpg_lag_2', 'gpg_lag_3', 'gpg_lag_4', 'gpg_lag_5', 'apg_lag_1',
            'xGoalsFor_team', 'goalsFor_team', 'highDangerShotsFor_team', 'highDangerxGoalsFor_team',
            'highDangerGoalsFor', 'xGoalsFor - goalsFor', 'games_played_per']

    df_2024_predict = df_2024[features]

    prediction = model.predict(df_2024_predict)

    df_2024["Predicted Goals Scored Percentage"] = prediction
    df_2024["Predicted Goals Scored per 82 Games"] = np.ceil(df_2024["Predicted Goals Scored Percentage"] * 82)

    feature_final = ["name", "position", "age", "games_played", "goals", "points", "pAssists", "sAssists", 
                "icetime_per_game", "ppg", "apg", "gpg", "goalsFor_team", "Predicted Goals Scored Percentage", 
                "Predicted Goals Scored per 82 Games"]

    df_2024 = df_2024[feature_final]
    df_2024.to_csv("../data/final/goals_pred.csv")

def processData():
    df = pd.read_csv("../data/cleaned_data.csv")
    df = df[df["season"] == 2024]

    return df

if __name__ == '__main__':
    predict()