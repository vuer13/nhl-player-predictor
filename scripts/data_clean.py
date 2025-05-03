import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from tqdm import tqdm

def get_df():
    # all data sets combined into one

    all_players = []

    for year in range(2015, 2025):
        file = f"data/pre-cleaned/players_{year}.csv"
        clean_data = clean_season_df(pd.read_csv(file))
        all_players.append(clean_data) 

    all_players_df = pd.concat(all_players).reset_index(drop=True)
    all_players_df.to_csv("data/combined_players_draft.csv", index=False)

    lag_predict_data(all_players_df)

    edit = pd.read_csv("data/combined_players_draft2.csv")

    edit = edit[edit["season"] > 2018]

    edit = edit.dropna(subset=[
        "next_assists_per_game",
        "next_goals_per_game",
    ])

    edit = addData(edit)

    edit.to_csv("data/combined_players_draft3.csv", index=False)

def lag_predict_data(all_players_df):
    to_lag = all_players_df
    to_lag = to_lag.sort_values(by=["playerId", "season"])

    to_lag_5 = ["points", "ppg", "pAssists", "sAssists", "goals", "gpg", "apg"]

    for stat in to_lag_5:
        for lag in range(1, 6):
            lag_col = f"{stat}_lag_{lag}"
            to_lag[lag_col] = to_lag.groupby("playerId")[stat].shift(lag)
            to_lag[f"{lag_col}_missing"] = to_lag[lag_col].isna().astype(int)
            to_lag[lag_col] = to_lag[lag_col].fillna(0)

    to_lag_1 = ["icetime_per_game"]

    lag_col = "icetime_per_game_lag_1"
    to_lag[lag_col] = to_lag.groupby("playerId")["icetime_per_game"].shift(lag)
    to_lag[f"{lag_col}_missing"] = to_lag[lag_col].isna().astype(int)
    to_lag[lag_col] = to_lag[lag_col].fillna(0)
    
    to_lag["next_goals_per_game"] = to_lag.groupby("playerId")["gpg"].shift(-1)
    to_lag["next_assists_per_game"] = to_lag.groupby("playerId")["apg"].shift(-1).values.flatten()

    to_lag.to_csv("data/combined_players_draft2.csv", index=False)

def clean_season_df(df):

    # to individually clean each dataset

    if "situation" in df.columns:
        df_filtered = df[df["situation"] == "all"].copy()

    columns = ["playerId", "season", "name", "position", "team", "games_played", "I_F_xGoals", "I_F_primaryAssists",
           "I_F_secondaryAssists", "I_F_shotsOnGoal", "I_F_points", "I_F_goals", "icetime",
           "OnIce_F_highDangerShots", "OnIce_F_goals"]

    df_filtered = df_filtered[columns]

    df_filtered["icetime_per_game"] = (df_filtered["icetime"] / 60) / df_filtered["games_played"]
    df_filtered["shot_percentage"] = (df_filtered["I_F_goals"] / df_filtered["I_F_shotsOnGoal"])
    df_filtered["points_per_60"] = (df_filtered["I_F_points"] / 
                                            (df_filtered["icetime_per_game"] * df_filtered["games_played"])) * 60
    df_filtered["ixG-goals"] = (df_filtered["I_F_xGoals"] - df_filtered["I_F_goals"])         
    df_filtered["ppg"] = (df_filtered["I_F_points"] / df_filtered["games_played"])
    df_filtered["apg"] = ((df_filtered["I_F_primaryAssists"] + df_filtered["I_F_secondaryAssists"])/ df_filtered["games_played"])
    df_filtered["gpg"] = (df_filtered["I_F_goals"] / df_filtered["games_played"])
    df_filtered = df_filtered.drop(["icetime"], axis = 1)

    df_filtered.rename(
        columns={
            "I_F_xGoals": "xGoals",
            "I_F_primaryAssists": "pAssists",
            "I_F_secondaryAssists": "sAssists",
            "I_F_shotsOnGoal": "sog",
            "I_F_points": "points",
            "I_F_goals": "goals",
            "OnIce_F_highDangerShots": "on_ice_chances",
            "OnIce_F_goals": "on_ice_goals"
        },
        inplace=True
    )

    return df_filtered

def addData(df):
    player_ids = df["playerId"].unique()

    players = []
    for p in tqdm(player_ids):
        data = get_player_info(p)
        if data:
            players.append(data)
        time.sleep(0.15)

    player_birthdates = pd.DataFrame(players)
    df_merged = df.merge(player_birthdates, on="playerId", how="left")
    df_merged["birthYear"] = pd.to_datetime(df_merged["birthdate"]).dt.year
    df_merged["age"] = df_merged["season"] - df_merged["birthYear"]
    df_merged = df_merged.drop(["birthdate", "birthYear"], axis = 1)

    return df_merged


def get_player_info(player_id):
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nhl.com/"
    }

    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()

        player = res.json()

        return {
            "playerId": player_id,
            "birthdate": player.get("birthDate")
        }

    except Exception as e:
        print(played_id + " " + res.status_code)
        return None

if __name__ == '__main__':
    get_df()
