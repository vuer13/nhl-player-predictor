import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.fc(hn[-1])
        return out

model = GRU(input_size=30,
             hidden_size=220, 
             num_layers=2,
             dropout=0.10551929610710889)
model.load_state_dict(torch.load('../models/gru_nhl_model.pt'))
model.eval()

X_scaler = joblib.load("../models/X_scaler_gru.pkl")
y_scaler = joblib.load("../models/y_scaler_gru.pkl")

df = pd.read_csv('../data/lstm_dataset_final.csv')

for col in ['pos_C', 'pos_D', 'pos_L', 'pos_R']:
    df[col] = df[col].astype(int)

features = [
    'xGoals', 'pAssists', 'sAssists', 'sog', 'points', 'goals',
    'on_ice_chances', 'on_ice_goals', 'hits', 'penality_mins', 'pen_drawn',
    'corsi', 'xGoalsForAfterShifts', 'corsiForAfterShifts',
    'icetime_per_game', 'shot_percentage', 'points_per_60',
    'ixG-goals', 'ppg', 'apg', 'gpg',
    'xGoalsFor-goalsFor_team', 'age', 'age2', 'games_played_per',
    'pos_C', 'pos_D', 'pos_L', 'pos_R'
]
targets = ['next_games_played_per', 'next_goals_per_game', 'next_assists_per_game']

df = df.dropna(subset=features)

df = df[df["season"] > 2021]

X_input = []
player_ids = []

grouped = df.groupby('playerId')

for player_id, group in grouped:
    group = group.sort_values('season')
    past = group[group['season'] > 2021]
    
    if len(past) == 0:
        continue
    
    sequence = past[features].values
    is_padded = np.zeros((3, 1)) 

    if sequence.shape[0] == 1:
        sequence = np.vstack([sequence[0], sequence[0], sequence[0]])
        is_padded[:2] = 1
    elif sequence.shape[0] == 2:
        sequence = np.vstack([sequence[0], sequence[0], sequence[1]])
        is_padded[:1] = 1
    elif sequence.shape[0] > 3:
        sequence = sequence[-3:]
        
    sequence = np.concatenate([sequence, is_padded], axis=1)
    X_input.append(sequence)
    player_ids.append(player_id)

X_input = np.array(X_input)

X_flat = X_input.reshape(-1, X_input.shape[2])
X_scaled = X_scaler.transform(X_flat)
X_scaled = X_scaled.reshape(X_input.shape)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

with torch.no_grad():
    preds_scaled = model(X_tensor).numpy()
    
preds = y_scaler.inverse_transform(preds_scaled)

df_preds = pd.DataFrame(preds, columns=[
    "pred_next_games_played_per",
    "pred_next_goals_per_game",
    "pred_next_assists_per_game"
])
df_preds["playerId"] = player_ids

df_preds.to_csv("../data/gru_predictions.csv", index=False)