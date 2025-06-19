import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import joblib
import optuna

def model():
    df = pd.read_csv("../data/cleaned_data_train.csv")

    features = ["age", "age2", 'pos_C', 'pos_D', 'pos_L', 'pos_R', 'games_played', 'xGoals',
            'pAssists', 'sAssists', 'sog', 'points', 'goals', 'on_ice_chances', 'on_ice_goals', 
            'icetime_per_game', 'shot_percentage', 'points_per_60', 'ixG-goals', 'ppg', 'apg', 'gpg',
            'points_lag_1', 'points_lag_2', 'points_lag_3', 'points_lag_4', 'points_lag_5', 
            'ppg_lag_1', 'ppg_lag_2', 'ppg_lag_3', 'ppg_lag_4', 'ppg_lag_5', 'pAssists_lag_1', 'sAssists_lag_1',
            'goals_lag_1', 'goals_lag_2', 'goals_lag_3', 'goals_lag_4', 'goals_lag_5', 
            'gpg_lag_1', 'gpg_lag_2', 'gpg_lag_3', 'gpg_lag_4', 'gpg_lag_5', 'apg_lag_1',
            'xGoalsFor_team', 'goalsFor_team', 'highDangerShotsFor_team', 'highDangerxGoalsFor_team',
            'highDangerGoalsFor', 'xGoalsFor - goalsFor', 'games_played_per', 'goals_weighted', 'gpg_weighted',
            'corsi', "xGoalsForAfterShifts", "corsiForAfterShifts"]
    target = "next_goals_per_game"

    df_model = df

    X = df_model[features]
    Y = df_model[target]

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    def estimate(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.3, 0.8),
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 4.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 8),
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
    print("Val   R²:", r2_score(Y_valid, val_preds))

    joblib.dump(xgb, "../models/xgb_model_goals.pkl")

if __name__ == '__main__':
    model()