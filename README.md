# NHL Player Points Predictor

This project utilizes 3 different XGBoost models to predict the percentage of games played in a season, goals per game and assists per game. The percentage of games played in a season was multiplied by 82 games, and the ceiling of that value was taken to be the predicted number of games played next season. That number was used to calculate the number of goals and assists the player would get. Those values would be added up to predict the number of points a player would achieve in the upcoming season. 

**Note: this only is for players who played at least 8 games this prior season**

Games played percentage model:
- MAE: 0.0554
- MSE: 0.1824
- R²: 0.4268

The models training and valdiation R² showed similar scores, indicating minimal overfitting. However, this model did score the lowest R² score and a somewhat high MSE score after tuning. Overall, this model doesn't do the best job in predicting the player's games played percentage. More advanced predictors would help enhance the model. However, with limited access to predictors related to the target, this was the best the model could do. 

Goals per game model:
- MAE: 0.05503
- MSE: 0.0056
- R²: 0.6866

This models training and validation R² score are the most similar, indicating very minimal overfitting (almost none). This model does obtain the best MSE and R² scores. Though we can improve the model with better predictors such as the player's linemates and other team factors, this model does do a great job predicting the goals scored for a given player.

Assists per game mode:
- MAE: 0.0804
- MSE: 0.0108
- R²: 0.6795

The models training and valdiation R² showed similar scores, indicating minimal overfitting. I noticed this model has difficulty predicting assists for players who score high values, so I specifically tuned it so players with a higher assist per game would be weighed more. The model can be improved with better predictors to enhance the model, but this model does the good enough job in predicting the assists scored for a given player.

# UPDATE: 06/2025
This project was further predicted with a LSTM and GRU model, using the pytorch library. The statistics for each model are as follows:

LSTM:

Games played percentage
- R²: 0.2228
- MAE: 0.2015
- MSE: 0.0726

Goals per game
- R²: 0.5612
- MAE: 0.0638
- MSE: 0.0089

Assists per game
- R²: 0.5774
- MAE: 0.0935
- MSE: 0.0157

GRU:

Games played percentage
- R²: 0.2484
- MAE: 0.1996
- MSE: 0.0703

Goals per game
- R²: 0.5640
- MAE: 0.0631
- MSE: 0.0089

Assists per game
- R²: 0.5958
- MAE: 0.0918
- MSE: 0.0150

Both models performed similarly, with the GRU model outperforming the LSTM model slightly. These two models outperformed the XGBoost model when predicting the games played percentage, but not the other two categories. The overall model scored a lower R² score compared to the XGBoost model but scored similar MSE scores.

# Next Step
- Creating a frontend for the statistics predicted