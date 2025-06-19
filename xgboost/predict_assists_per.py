import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def predict():
    df_2024 = processData()
    model = joblib.load('../models/xgb_model_assists.pkl')

    features = ["age", "age2", 'pos_C', 'pos_D', 'pos_L', 'pos_R', 'games_played',
            'pAssists', 'sAssists', 'sog', 'points', 'goals', 'on_ice_chances', 'on_ice_goals', 
            'icetime_per_game', 'shot_percentage', 'points_per_60', 'ppg', 'apg', 'gpg',
            'points_lag_1', 'points_lag_2', 'points_lag_3', 'points_lag_4', 'points_lag_5', 
            'ppg_lag_1', 'ppg_lag_2', 'ppg_lag_3', 'ppg_lag_4', 'ppg_lag_5', 'gpg_lag_1', 
            'pAssists_lag_1', 'pAssists_lag_2', 'pAssists_lag_3', 'pAssists_lag_4', 'pAssists_lag_5', 
            'sAssists_lag_1', 'sAssists_lag_2', 'sAssists_lag_3', 'sAssists_lag_4', 'sAssists_lag_5',
            'apg_lag_1', 'apg_lag_2', 'apg_lag_3', 'apg_lag_4', 'apg_lag_5',
            'xGoalsFor_team', 'goalsFor_team', 'highDangerShotsFor_team', 'highDangerxGoalsFor_team',
            'highDangerGoalsFor', 'xGoalsFor - goalsFor', 'games_played_per', 'assists_weighted', 'apg_weighted',
            'corsi', "xGoalsForAfterShifts", "corsiForAfterShifts"]

    df_2024_predict = df_2024[features]

    prediction = model.predict(df_2024_predict)

    df_2024["Predicted Assists Percentage"] = np.expm1(prediction)
    df_2024["Predicted Assists per 82 Games"] = np.ceil(df_2024["Predicted Assists Percentage"] * 82)

    feature_final = ["name", "playerId", "position", "age", "games_played", "goals", "points", "pAssists", "sAssists", 
                "icetime_per_game", "ppg", "apg", "gpg", "goalsFor_team", "Predicted Assists Percentage", 
                "Predicted Assists per 82 Games"]

    df_2024 = df_2024[feature_final]
    df_2024.to_csv("../data/final/assists_pred.csv")

def processData():
    df = pd.read_csv("../data/cleaned_data.csv")
    df = df[df["season"] == 2024]

    return df

if __name__ == '__main__':
    predict()