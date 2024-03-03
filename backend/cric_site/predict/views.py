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





# Load model
model = MyLSTMWithSoftmax(99, 64, 8)
state_dict = torch.load("static_folder/cric_model.pth")
model.load_state_dict(state_dict)

# Importing Prediction Data
import pandas as pd
df = pd.read_csv('static_folder/cleaned_stats_data.csv')

# Retrieve unique IDs from the 'ID' column of the DataFrame
unique_ids = df['ID'].unique()

# Initialize an empty dictionary to store DataFrames corresponding to each unique ID
id_dataframes = {}
for unique_id in unique_ids:
    # Create a DataFrame containing rows where the 'ID' column matches the current unique ID
    id_dataframes[unique_id] = df[df['ID'] == unique_id]
selected_match_df = id_dataframes[1312199]
selected_match_df.head(5)
features = selected_match_df[['innings', 'overs', 'ballnumber', 'batsman_run', 'extras_run', 'total_run',
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
features_tensor = torch.tensor(features.values, dtype=torch.float32)
features_tensor = features_tensor.unsqueeze(0)

# Create a DataLoader with batch size 1
single_match_loader = DataLoader(TensorDataset(features_tensor), batch_size=1, shuffle=False)
class_mapping = {0: 'delivery_type_0', 1: 'delivery_type_1', 2: 'delivery_type_2',
                 3: 'delivery_type_3',
                 4: 'delivery_type_4',5: 'delivery_type_5',
                 6: 'delivery_type_6',7: 'delivery_type_7'}






predictions = []
previous_prediction = None

model.eval()

# Assuming features_tensor contains the entire match data
for i in range(features_tensor.size(1)):  # Iterate through each ball in the match
    current_input = features_tensor[:, i:i+1, :]

    if previous_prediction is not None:
        # Update the input tensor with the previous prediction
        current_input[0, 0, -1] = previous_prediction.item()

    with torch.no_grad():
        out = model(current_input)
        y_pred = F.softmax(out, dim=1)
        our_prediction = torch.argmax(y_pred, dim=1).item()

    # Save the prediction and update the previous prediction for the next iteration
    predictions.append(our_prediction)
    previous_prediction = torch.tensor(our_prediction, dtype=torch.float32)



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

delivery_type_columns = ['delivery_type_0', 'delivery_type_1', 'delivery_type_2', 'delivery_type_3', 'delivery_type_4', 'delivery_type_5', 'delivery_type_6', 'delivery_type_7']

predicted_selected_match_df['actual_outcome'] = (
    predicted_selected_match_df[delivery_type_columns].apply(lambda row: sum(row[col] * column_mapping[col] for col in delivery_type_columns), axis=1)
)
predicted_selected_match_df.drop('actual_outcome',axis=1,inplace=True)
predicted_selected_match_df['predicted_current_score'] = predicted_selected_match_df.groupby(['ID', 'innings'])['predicted_outcome'].cumsum()

# Reset the index if needed
predicted_selected_match_df = predicted_selected_match_df.reset_index(drop=True)

class PredictionAPIView(APIView):
    def get(self, request):
        # Serialize the DataFrame to JSON
        df_json = predicted_selected_match_df.to_json(orient='records')
        
        # Include the serialized DataFrame in the response data
        response_data = {
            'data': df_json
        }
        
        return Response(response_data)

