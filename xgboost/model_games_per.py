import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import optuna
import joblib

def model():
    df = pd.read_csv("../data/cleaned_data_train.csv")

    features = ["games_played", "icetime_per_game", "icetime_per_game_lag_1", "games_played_per", "games_played_per_lag_1",
            "games_played_per_lag_2", "games_played_per_lag_3", "games_played_per_lag_4", "games_played_per_lag_5",
            "games_played_lag_1", "points_per_60", "age", "age2", 'pos_C', 'pos_D', 'pos_L', 'pos_R', 'points', 'on_ice_chances',
            'hits', 'penality_mins', 'pen_drawn', 'corsi', "games_played_lag_2", "games_played_lag_3", 'on_ice_goals',
            "xGoalsForAfterShifts", "corsiForAfterShifts", "iceTimeRank"]
    target = "next_games_played_per"

    df_model = df

    X = df_model[features]
    Y = df_model[target]

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    def estimate(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
            "random_state": 0
        }

        model = XGBRegressor(**params)
        model.fit(X_train, Y_train)

        preds = model.predict(X_valid)
        return r2_score(Y_valid, preds)

    study = optuna.create_study(direction = "maximize")
    study.optimize(estimate, n_trials = 150)

    xgb = XGBRegressor(**study.best_trial.params)
    xgb.fit(X_train, Y_train)

    y_pred = xgb.predict(X_valid)
    mse = mean_squared_error(Y_valid, y_pred)
    mae= mean_absolute_error(Y_valid, y_pred)
    r2 = r2_score(Y_valid, y_pred)

    print("MSE:", mse)
    print("MAE:", mae)
    print("R²:", r2)

    # Check overfitting
    train_preds = xgb.predict(X_train)
    val_preds = xgb.predict(X_valid)

    print("Train R²:", r2_score(Y_train, train_preds))
    print("Val R²:", r2_score(Y_valid, val_preds))

    joblib.dump(xgb, "../models/xgb_model_games_played.pkl")

if __name__ == '__main__':
    model()
