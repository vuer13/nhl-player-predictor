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
import shap

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
    xgb = pipe.named_steps["xgbregressor"]
    feature_names = pipe.named_steps["columntransformer"].get_feature_names_out()

    imp = pd.Series(
        xgb.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print(imp.head(25))
    return imp

    
def run_shap(pipe, X_valid):
    X_val_trans = pipe.named_steps["columntransformer"].transform(X_valid)

    explainer = shap.TreeExplainer(
        pipe.named_steps["xgbregressor"]
    )

    shap_values = explainer.shap_values(X_val_trans)

    feature_names = pipe.named_steps["columntransformer"].get_feature_names_out()

    shap.summary_plot(
        shap_values,
        X_val_trans,
        feature_names=feature_names
    )


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
        
    df_model = df

    X = df[features]
    y = df["log_apg"]
    
    sample_weight = df["apg"].apply(lambda x: 6.25 if x > 0.9 else 1.0)
    
    X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
        X, y, sample_weight,
        test_size=0.2,
        random_state=42
    )

    pipe = build_pipeline(X)

    def estimate(trial):
        params = {
            "xgbregressor__n_estimators": trial.suggest_int("n_estimators", 300, 700),
            "xgbregressor__max_depth": trial.suggest_int("max_depth", 2, 3),
            "xgbregressor__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "xgbregressor__subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "xgbregressor__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "xgbregressor__reg_alpha": trial.suggest_float("reg_alpha", 1.5, 3.0),
            "xgbregressor__reg_lambda": trial.suggest_float("reg_lambda", 3.0, 9.0),
            "xgbregressor__min_child_weight": trial.suggest_int("min_child_weight", 8, 10),
        }       

        pipeline.set_params(**params)

        pipe.fit(
            X_train,
            y_train,
            xgbregressor__sample_weight=w_train
        )

        preds = pipeline.predict(X_valid)
        return r2_score(y_valid, preds)

    study = optuna.create_study(direction = "maximize")
    study.optimize(estimate, n_trials = 150)

    pipe.set_params(**study.best_params)
    pipe.fit(X_train, y_train, xgbregressor__sample_weight=w_train)
    
    log_preds = pipe.predict(X_valid)

    y_pred = np.expm1(log_preds)
    y_true = np.expm1(y_valid)

    print("MSE:", mean_squared_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R²:", r2_score(y_true, y_pred))

    print("Train R² (log):", r2_score(y_train, pipe.predict(X_train)))
    print("Val R² (log):", r2_score(y_valid, pipe.predict(X_valid)))
    
    feature_importance(pipe)
    run_shap(pipe, X_valid)

    joblib.dump(pipe, "../models/xgb_model_assists.pkl")
    print("Model Saved")

if __name__ == '__main__':
    model()