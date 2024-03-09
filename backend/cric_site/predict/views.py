# views.py
from rest_framework.views import APIView, View
from rest_framework.response import Response
from rest_framework import status
from predict.models import MyLSTMWithSoftmax
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import json
import numpy as np


# Load model
model = MyLSTMWithSoftmax(96, 64, 8)
state_dict = torch.load("static_folder/cric_model_2nd.pth")
model.load_state_dict(state_dict)

# Importing Prediction Data
import pandas as pd
df = pd.read_csv('static_folder/final_stats_data.csv', low_memory=False)
df['delivery_type'] = df['total_run'].astype(str)

#Remove unnecessary data
clean_df = df.copy()
clean_df.drop(['non-striker', 'extra_type',
       'non_boundary', 'player_out', 'kind',
       'fielders_involved', 'City','MatchNumber','SuperOver',
       'WonBy', 'Margin', 'method','Player_of_Match',
       'Team1Players', 'Team2Players', 'Umpire1', 'Umpire2',
       'WinningTeam', 'Team2','Date','Team1','Venue','TossWinner','batter','bowler'], axis = 1, inplace = True)

from sklearn.preprocessing import OneHotEncoder

# Assuming 'df' is your DataFrame
columns_to_encode = ['Season', 'BattingTeam', 'BowlingTeam','delivery_type','TossDecision']

# Initialize the encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform on the training data
encoder.fit(clean_df[columns_to_encode])

# Transform the specified categorical columns to one-hot encoded representation
one_hot_encoded = encoder.transform(clean_df[columns_to_encode])

# Concatenate the one-hot encoded features with the original DataFrame
clean_df = pd.concat([clean_df, pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))], axis=1)

# Drop the original categorical columns
clean_df = clean_df.drop(columns_to_encode, axis=1)

# Retrieve unique IDs from the 'ID' column of the DataFrame
unique_ids = clean_df['ID'].unique()

# Specify the columns to scale
columns_to_scale = ['strike_rate_x', 'batting_average', 'strike_rate_y', 'bowling_average',
                    'economy','runs_conceded','runs_scored','balls_faced','balls_bowled',
                    'batter_matches_played','0s_scored', '1s_scored', '2s_scored', '4s_scored', '6s_scored',
                     'high_score', '25_scored', '50_scored', '75_scored', '100_scored','0_wickets_taken', '1_wickets_taken',
                    '2_wickets_taken', '3_wickets_taken', '4_wickets_taken', '5_wickets_taken','6_wickets_taken',
                    'bowler_matches_played','wickets_taken','4s_conceded','6s_conceded', '0s_conceded', '1s_conceded',
                    '2s_conceded', 'highest_conceded',]

# Replace infinite or too large values with the median
clean_df[columns_to_scale] = clean_df[columns_to_scale].replace([np.inf, -np.inf], np.nan)
clean_df[columns_to_scale] = clean_df[columns_to_scale].fillna(clean_df[columns_to_scale].median())

from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
clean_df[columns_to_scale] = scaler.fit_transform(clean_df[columns_to_scale])



# Initialize an empty dictionary to store DataFrames corresponding to each unique ID
id_dataframes = {}
unique_ids = clean_df['ID'].unique()
for unique_id in unique_ids:
    # Create a DataFrame containing rows where the 'ID' column matches the current unique ID
    id_dataframes[unique_id] = clean_df[clean_df['ID'] == unique_id]
selected_match_id = 1304061
selected_match_df = id_dataframes[selected_match_id]
features = selected_match_df[['innings', 'overs', 'ballnumber',
                               'isWicketDelivery', 'batter_matches_played', 'runs_scored', 'dismissals',
                               'balls_faced', '0s_scored', '1s_scored', '2s_scored', '4s_scored', '6s_scored',
                               'high_score', '25_scored', '50_scored', '75_scored', '100_scored', 'strike_rate_x',
                               'batting_average', 'notout', 'explosivity_rating', '0_wickets_taken', '1_wickets_taken',
                               '2_wickets_taken', '3_wickets_taken', '4_wickets_taken', '5_wickets_taken',
                               '6_wickets_taken', 'bowler_matches_played', 'runs_conceded', 'extras_runs_conceded',
                               'wickets_taken', 'balls_bowled', '4s_conceded', '6s_conceded', '0s_conceded',
                               '1s_conceded', '2s_conceded', 'highest_conceded', 'strike_rate_y', 'bowling_average',
                               'economy', 'total_runs_conceded', 'target', 'current_score', 'balls_left',
                               'wickets_left', 'runs_left', 'Season_2008', 'Season_2009', 'Season_2010', 'Season_2011',
                               'Season_2012', 'Season_2013', 'Season_2014', 'Season_2015', 'Season_2016', 'Season_2017',
                               'Season_2018', 'Season_2019', 'Season_2020', 'Season_2021', 'Season_2022',
                               'BattingTeam_Chennai Super Kings', 'BattingTeam_Deccan Chargers', 'BattingTeam_Delhi Capitals',
                               'BattingTeam_Gujarat Lions', 'BattingTeam_Gujarat Titans', 'BattingTeam_Kochi Tuskers Kerala',
                               'BattingTeam_Kolkata Knight Riders', 'BattingTeam_Lucknow Super Giants', 'BattingTeam_Mumbai Indians',
                               'BattingTeam_Pune Warriors', 'BattingTeam_Punjab Kings', 'BattingTeam_Rajasthan Royals',
                               'BattingTeam_Rising Pune Supergiant', 'BattingTeam_Royal Challengers Bangalore',
                               'BattingTeam_Sunrisers Hyderabad', 'BowlingTeam_Chennai Super Kings', 'BowlingTeam_Deccan Chargers',
                               'BowlingTeam_Delhi Capitals', 'BowlingTeam_Gujarat Lions', 'BowlingTeam_Gujarat Titans',
                               'BowlingTeam_Kochi Tuskers Kerala', 'BowlingTeam_Kolkata Knight Riders', 'BowlingTeam_Lucknow Super Giants',
                               'BowlingTeam_Mumbai Indians', 'BowlingTeam_Pune Warriors', 'BowlingTeam_Punjab Kings',
                               'BowlingTeam_Rajasthan Royals', 'BowlingTeam_Rising Pune Supergiant',
                               'BowlingTeam_Royal Challengers Bangalore', 'BowlingTeam_Sunrisers Hyderabad',
                               'TossDecision_bat', 'TossDecision_field']]
# Convert to PyTorch tensor
import numpy as np
features_tensor = torch.tensor(features.to_numpy().reshape(1,-1,96).astype(np.float32))



model.eval()

import random
predictions = []

# Assuming features_tensor contains the entire match data
for i in range(features_tensor.size(1)):  # Iterate through each ball in the match
    current_input = features_tensor[:, i:i+6, :]
    with torch.no_grad():
        out = model(current_input)
        y_pred = F.softmax(out, dim=1)
        our_predictions = torch.argmax(y_pred, dim=1).item()
        if our_predictions in [4, 6]:
            if random.random() < 0.6:
                our_predictions -= 2
        if our_predictions in [0]:
            if random.random() < 0.6:
                our_predictions += 1
    predictions.append(our_predictions)
    
    


predicted_selected_match_df = selected_match_df.copy()
predicted_selected_match_df['predicted_outcome'] = predictions

column_mapping = {
'delivery_type_0': 0,
'delivery_type_1': 1,
'delivery_type_2': 2,
'delivery_type_3': 3,
'delivery_type_4': 4,
'delivery_type_5': 5,
'delivery_type_6': 6,
'delivery_type_7': 7,
}
delivery_type_columns = ['delivery_type_0', 'delivery_type_1', 'delivery_type_2',
'delivery_type_3', 'delivery_type_4',
'delivery_type_5', 'delivery_type_6', 'delivery_type_7']
predicted_selected_match_df['actual_outcome'] = (
    predicted_selected_match_df[delivery_type_columns].apply(lambda row: sum(row[col] * column_mapping[col] for col in delivery_type_columns), axis=1)
)

predicted_selected_match_df['predicted_current_score'] = predicted_selected_match_df.groupby(['ID', 'innings'])['predicted_outcome'].cumsum()

# Reset the index if needed

predicted_selected_match_df = predicted_selected_match_df.reset_index(drop=True)

predicted_selected_match_df['current_ball_number'] = predicted_selected_match_df.groupby('innings').cumcount()








class PredictionAPIView(APIView):
    def get(self, request):
        # Group the DataFrame by specific columns to organize the data
        grouped = predicted_selected_match_df.groupby(['ID', 'innings', 'overs'])

        # Dictionary to store the renamed match IDs and their corresponding data
        renamed_result = {}

        # Iterate through each group
        for name, group in grouped:
            idx, inning, over = name
            match_data = group.to_dict(orient='records')

            # Extract batting and bowling team names
            batting_team = None
            bowling_team = None

            for data in match_data:
                for key, value in data.items():
                    if key.startswith('BattingTeam_') and value == 1.0:
                        batting_team = key.split('_')[1]
                    elif key.startswith('BowlingTeam_') and value == 1.0:
                        bowling_team = key.split('_')[1]

            # Construct the renamed match ID if both teams are found
            if batting_team and bowling_team:
                renamed_match_id = f"{batting_team} vs {bowling_team}"
                renamed_result[renamed_match_id] = match_data

        return Response(renamed_result, status=status.HTTP_200_OK)
    


class PlotDataAPIView(APIView):
    def get(self, request):
        
        predicted_selected_match_df = selected_match_df.copy()
        predicted_selected_match_df['predicted_outcome'] = predictions

        column_mapping = {
        'delivery_type_0': 0,
        'delivery_type_1': 1,
        'delivery_type_2': 2,
        'delivery_type_3': 3,
        'delivery_type_4': 4,
        'delivery_type_5': 5,
        'delivery_type_6': 6,
        'delivery_type_7': 7,
        }
        delivery_type_columns = ['delivery_type_0', 'delivery_type_1', 'delivery_type_2',
        'delivery_type_3', 'delivery_type_4',
        'delivery_type_5', 'delivery_type_6', 'delivery_type_7']
        predicted_selected_match_df['actual_outcome'] = (
            predicted_selected_match_df[delivery_type_columns].apply(lambda row: sum(row[col] * column_mapping[col] for col in delivery_type_columns), axis=1)
        )

        predicted_selected_match_df['predicted_current_score'] = predicted_selected_match_df.groupby(['ID', 'innings'])['predicted_outcome'].cumsum()

        # Reset the index if needed

        predicted_selected_match_df = predicted_selected_match_df.reset_index(drop=True)

        predicted_selected_match_df['current_ball_number'] = predicted_selected_match_df.groupby('innings').cumcount()

        column_mapping = {
            'delivery_type_0': 0,
            'delivery_type_1': 1,
            'delivery_type_2': 2,
            'delivery_type_3': 3,
            'delivery_type_4': 4,
            'delivery_type_5': 5,
            'delivery_type_6': 6,
            'delivery_type_7': 7,
        }
        delivery_type_columns = ['delivery_type_0', 'delivery_type_1', 'delivery_type_2',
                                'delivery_type_3', 'delivery_type_4',
                                'delivery_type_5', 'delivery_type_6', 'delivery_type_7']
        predicted_selected_match_df['actual_outcome'] = (
            predicted_selected_match_df[delivery_type_columns].apply(lambda row: sum(row[col] * column_mapping[col] for col in delivery_type_columns), axis=1)
        )

        predicted_selected_match_df['predicted_current_score'] = predicted_selected_match_df.groupby(['ID', 'innings'])['predicted_outcome'].cumsum()

        predicted_selected_match_df = predicted_selected_match_df.reset_index(drop=True)

        predicted_selected_match_df['current_ball_number'] = predicted_selected_match_df.groupby('innings').cumcount()

        # Separate data for each inning
        inning1_data = predicted_selected_match_df[predicted_selected_match_df['innings'] == 1]
        inning2_data = predicted_selected_match_df[predicted_selected_match_df['innings'] == 2]

        inning1_actual_total_run = inning1_data['actual_outcome'].sum()
        inning1_predicted_total_run = inning1_data['predicted_outcome'].sum()

        inning2_actual_total_run = inning2_data['actual_outcome'].sum()
        inning2_predicted_total_run = inning2_data['predicted_outcome'].sum()

        # Read final stats data
        m_df = df.copy()
        m_df = m_df[m_df['ID'] == selected_match_id]
        m_inning1_data = m_df[m_df['innings'] == 1]
        m_inning2_data = m_df[m_df['innings'] == 2]
        actual_winning_team = list(m_df['WinningTeam'].unique())[0]
        team_1 = list(m_inning1_data['BattingTeam'].unique())[0]
        team_2 = list(m_inning2_data['BattingTeam'].unique())[0]
        predicted_winning_team = team_1 if inning1_predicted_total_run > inning2_predicted_total_run else team_2
        
        inning1_data['cumulative_actual_runs'] = inning1_data['actual_outcome'].cumsum()
        inning2_data['cumulative_actual_runs'] = inning2_data['actual_outcome'].cumsum()
        inning1_data['cumulative_predicted_runs'] = inning1_data['predicted_current_score']
        inning2_data['cumulative_predicted_runs'] = inning2_data['predicted_current_score']
        
        inning1_data_json = inning1_data[['current_ball_number', 'cumulative_actual_runs', 'predicted_current_score']].to_dict(orient='records')
        inning2_data_json = inning2_data[['current_ball_number', 'cumulative_actual_runs', 'predicted_current_score']].to_dict(orient='records')

        # Prepare data for sending to frontend
        response_data = {
            'inning1_data': inning1_data_json,
            'inning2_data': inning2_data_json,
            'inning1_actual_total_run': inning1_actual_total_run,
            'inning1_predicted_total_run': inning1_predicted_total_run,
            'inning2_actual_total_run': inning2_actual_total_run,
            'inning2_predicted_total_run': inning2_predicted_total_run,
            'team_1': team_1,
            'team_2': team_2,
            'actual_winning_team': actual_winning_team,
            'predicted_winning_team': predicted_winning_team
        }

        return Response(response_data, status=status.HTTP_200_OK)
    

