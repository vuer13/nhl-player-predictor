import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import optuna
import joblib

# Trimmed lags incase 4/5 don't do anything
'''
"points_lag_1", "points_lag_2", "points_lag_3",
"ppg_lag_1", "ppg_lag_2", "ppg_lag_3",
"apg_lag_1", "apg_lag_2", "apg_lag_3",
"pAssists_lag_1", "pAssists_lag_2", "pAssists_lag_3",
"sAssists_lag_1", "sAssists_lag_2", "sAssists_lag_3",
'''

def preprocessing(X):
    """Preprocessing pipeline for the model."""
    numeric_features = X.columns.tolist()

    preprocessor = make_column_transformer(
        (make_pipeline(
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()))), numeric_features)

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=0,
        n_jobs=-1
    )

    pipe = make_pipeline(
        preprocessor,
        base_model
    )
    
    return pipe

def feature_importance(pipe):
    xgb_model = pipe.named_steps["xgbregressor"]
    feature_names = pipe.named_steps["columntransformer"].get_feature_names_out()
    importances = xgb_model.feature_importances_

    feat_imp = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
    )

    print(feat_imp)

def model():
    # Read csv file
    df = pd.read_csv("../data/cleaned_data_train.csv")

    df["log_apg"] = np.log1p(df["next_assists_per_game"])  

    features = [
        "age",
        "pos_C", "pos_D", "pos_L", "pos_R",
        "games_played",
        "pAssists", "sAssists", "sog", "points", "goals",
        "on_ice_chances", "on_ice_goals",
        "icetime_per_game", "shot_percentage",
        "points_per_60", "ppg", "apg", "gpg",

        # lagged stats
        'points_lag_1', 'points_lag_2', 'points_lag_3', 'points_lag_4', 'points_lag_5', 
        'ppg_lag_1', 'ppg_lag_2', 'ppg_lag_3', 'ppg_lag_4', 'ppg_lag_5', 'pAssists_lag_1', 'sAssists_lag_1',
        'goals_lag_1', 'goals_lag_2', 'goals_lag_3', 'goals_lag_4', 'goals_lag_5', 
        'gpg_lag_1', 'gpg_lag_2', 'gpg_lag_3', 'gpg_lag_4', 'gpg_lag_5', 'apg_lag_1',

        # team + context
        "xGoalsFor_team", "goalsFor_team",
        "highDangerShotsFor_team", "highDangerxGoalsFor_team",
        "xGoalsFor - goalsFor",
        "games_played_per",

        # engineered
        "assists_weighted", "apg_weighted",
        "corsi", "xGoalsForAfterShifts", "corsiForAfterShifts", "age2"
    ]
        
    target = "log_apg"

    df_model = df

    sample_weight = df_model["apg"].apply(lambda x: 6.25 if x > 0.9 else 1.0)

    X = df_model[features]
    Y = df_model[target]

    X_train, X_valid, Y_train, Y_valid, w_train, w_valid = train_test_split(X, Y, sample_weight, test_size=0.2, random_state=42)
    
    pipeline = preprocessing(X)

    def estimate(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 700),
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.5, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 3.0, 9.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 8, 10),
            "random_state": 0
        }

        pipeline.set_params(**params)

        pipeline.fit(
            X_train,
            y_train,
            xgbregressor__sample_weight=w_train
        )

        preds = pipeline.predict(X_valid)
        return r2_score(y_valid, preds)

    study = optuna.create_study(direction = "maximize")
    study.optimize(estimate, n_trials = 150)

    pipeline.set_params(**study.best_params)
    pipeline.fit(X_train, y_train, xgbregressor__sample_weight=w_train)
    
    feature_importance(pipeline)

    log_preds = pipe.predict(X_valid)

    y_pred = np.expm1(log_preds)
    y_true = np.expm1(y_valid)

    print("MSE:", mean_squared_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R²:", r2_score(y_true, y_pred))

    print("Train R² (log):", r2_score(y_train, pipe.predict(X_train)))
    print("Val R² (log):", r2_score(y_valid, pipe.predict(X_valid)))

    joblib.dump(xgb, "../models/xgb_model_assists.pkl")
    print("Model Saved")

if __name__ == '__main__':
    model()