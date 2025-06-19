import pandas as pd
import requests
import time
from datetime import datetime
from tqdm import tqdm

def main():
    goals = pd.read_csv("../data/final/goals_pred.csv")
    assists = pd.read_csv("../data/final/assists_pred.csv")
    games = pd.read_csv("../data/final/games_played_pred.csv")

    goals = goals[["playerId", "Predicted Goals Scored Percentage", "Predicted Goals Scored per 82 Games"]]
    assists = assists[["playerId", "Predicted Assists Percentage" , "Predicted Assists per 82 Games"]]

    goals_assists = goals.merge(assists, on = "playerId")

    full_df = games.merge(goals_assists, on = "playerId")

    player_ids = full_df["playerId"].unique()

    players = []
    for p in tqdm(player_ids):
        data = get_pictures(p)
        if data:
            players.append(data)
        time.sleep(0.10)

    headshots = pd.DataFrame(players)

    final_df = full_df.merge(headshots, on = 'playerId')

    final_df.to_csv("../data/final/final_df.csv")

def get_pictures(player_id):
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
            "headshot": player.get("headshot")
        }

    except Exception as e:
        print(player_id + " " + res.status_code)
        return None

if __name__ == '__main__':
    main()