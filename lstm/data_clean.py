import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from tqdm import tqdm

def addData():
    df = pd.read_csv('../data/combined_players_draft.csv')
    
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

    df_merged.to_csv('../data/lstm_dataset.csv')
    
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
    
def teamClean():
    all_teams = []

    for year in range(2015, 2025):
        file = f"../data/pre-cleaned/team_stats_{year}.csv"
        clean_data = pd.read_csv(file)
        all_teams.append(clean_data) 

    df = pd.concat(all_teams).reset_index(drop=True)
    
    if "situation" in df.columns:
        df_filtered = df[df["situation"] == "all"].copy()

    columns = ["team", "season", "games_played", "xGoalsFor", "goalsFor", "highDangerShotsFor", "highDangerxGoalsFor", "highDangerGoalsFor"]
    df_filtered = df_filtered[columns]
    df_filtered["xGoalsFor-goalsFor_team"] = df_filtered["xGoalsFor"] - df_filtered["goalsFor"]

    df_filtered.rename(
        columns={
            "games_played": "games_played_team",
            "goalsFor": "goalsFor_team",
            "highDangerShotsFor": "highDangerShotsFor_team",
            "highDangerxGoalsFor": "highDangerxGoalsFor_team",
            "xGoalsFor": "xGoalsFor_team"
        },
        inplace=True
    ) 
    
    df_players = pd.read_csv('../data/lstm_dataset.csv')
    
    df_merged_total = df_players.merge(df_filtered, on=["team", "season"], how="right")
    df_merged_total["age2"] = df_merged_total["age"] * df_merged_total["age"]
    df_merged_total["games_played_per"] = np.minimum(df_merged_total["games_played"] / df_merged_total["games_played_team"], 1.0)
    
    dummies = pd.get_dummies(df_merged_total["position"], prefix="pos", dummy_na=False)
    df_merged_total = pd.concat([df_merged_total, dummies], axis=1)
    
    df_merged_total["next_games_played_per"] = df_merged_total.groupby("playerId")["games_played_per"].shift(-1)
    df_merged_total["next_goals_per_game"] = df_merged_total.groupby("playerId")["gpg"].shift(-1)
    df_merged_total["next_assists_per_game"] = df_merged_total.groupby("playerId")["apg"].shift(-1).values.flatten()
    
    df_merged_total.to_csv('../data/lstm_dataset_final.csv')

    
if __name__ == '__main__':
    teamClean()