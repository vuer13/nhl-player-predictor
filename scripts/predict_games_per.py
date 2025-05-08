import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def predict():
    df_2024 = processData()
    model = joblib.load('../models/xgb_model_games_played.pkl')

    features = ["games_played", "icetime_per_game", "icetime_per_game_lag_1", "games_played_per", "games_played_per_lag_1",
            "games_played_per_lag_2", "games_played_per_lag_3", "games_played_per_lag_4", "games_played_per_lag_5",
            "games_played_lag_1", "points_per_60", "age", "age2", 'pos_C', 'pos_D', 'pos_L', 'pos_R']

    df_2024_predict = df_2024[features]

    prediction = model.predict(df_2024_predict)

    df_2024["Predicted Games Played Percentage"] = prediction
    df_2024["Predicted Games Played"] = np.minimum(np.ceil(df_2024["Predicted Games Played Percentage"] * 82), 82)

    feature_final = ["name", "position", "age", "games_played", "goals", "points", "pAssists", "sAssists", 
                "icetime_per_game", "ppg", "apg", "gpg", "goalsFor_team", "Predicted Games Played Percentage", 
                "Predicted Games Played"]

    df_2024 = df_2024[feature_final]
    df_2024.to_csv("../data/final/games_played_pred.csv")

def processData():
    df = pd.read_csv("../data/cleaned_data.csv")
    df = df[df["season"] == 2024]

    return df

if __name__ == '__main__':
    predict()