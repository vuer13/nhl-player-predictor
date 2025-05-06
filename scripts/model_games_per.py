import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import joblib

df = pd.read_csv("../data/cleaned_data_train.csv")

features = ["games_played", "icetime_per_game", "position", "icetime_per_game_lag_1",
            "games_played_per", "games_played_per_lag_1",
            "games_played_per_lag_2", "games_played_per_lag_3", "games_played_per_lag_4", "games_played_per_lag_5",
            "games_played_lag_1", "points_per_60", "age"]
target = ["next_games_played_per"]

df_model = df.dropna(subset = features + target)

X = df_model[features]
Y = df_model[target]

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 13)
X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape

# params from Jupyter Notebook
xgb = XGBRegressor(random_state = 13, 
                    verbosity = 0,
                    colsample_bytree = 0.8, 
                    learning_rate = 0.01, 
                    max_depth = 3, 
                    n_estimators = 300, 
                    reg_alpha = 0, 
                    reg_lambda = 5, 
                    subsample = 0.8)

le = LabelEncoder()
X_train["position"] = le.fit_transform(X_train["position"])
X_valid["position"] = le.transform(X_valid["position"])

xgb.fit(X_train, Y_train)

y_pred = xgb.predict(X_valid)
mse = mean_squared_error(Y_valid, y_pred)
mae= mean_absolute_error(Y_valid, y_pred)
r2 = r2_score(Y_valid, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)

joblib.dump(xgb, "../models/xgb_model_games_played.pkl")
