import pandas as pd
import numpy as np
import itertools
from flask import Flask, jsonify
app = Flask(__name__)

balls = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
unique_match_dataframes = {}
for unique_id in balls['ID'].unique():
    unique_match_dataframes[unique_id] = balls[balls['ID'] == unique_id].copy()
class Ball:
    def __init__(self, inning, over, ball):
        assert inning == 1 or inning == 2, "Inning wrong!"
        assert over <= 19 and over >= 0, "Over wrong!"
        assert ball <= 6 and ball >= 1, "Ball wrong!"

        self.inning = inning
        self.over = over
        self.ball = ball
        self.distribution = {-1:0, 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}

    def __repr__(self):
        return f"Inning: {self.inning}, Over: {self.over}, Ball: {self.ball}\n Distribution:{self.distribution}"


class Match:
    def __init__(self, team1='', team2='', stadium=''):
        pass  # Initialization function, not used in the provided code

    @staticmethod
    def random_simulate():
        # Create a dictionary 'balls' to store Ball objects for each possible combination of innings, overs, and ball number
        balls = {(i, o, b): Ball(i, o, b) for i, o, b in itertools.product(range(1, 3), range(0, 20), range(1, 7))}

        # Iterate through unique match dataframes (not provided in the code)
        for id, match in unique_match_dataframes.items():
            for i, r in match[['innings', 'overs', 'ballnumber', 'total_run', 'isWicketDelivery']].iterrows():

                # Ignore super overs
                if r['innings'] >= 3:
                    continue
                # Ignore ball number increasing due to wide ball
                if r['ballnumber'] >= 7:
                    continue

                # Update Ball objects based on whether it's a wicket delivery or not
                if r['isWicketDelivery']:
                    balls[(r['innings'], r['overs'], r['ballnumber'])].distribution[-1] += 1
                    continue
                balls[(r['innings'], r['overs'], r['ballnumber'])].distribution[r['total_run']] += 1

        iobs = []
        run_history = []
        for iob, B in balls.items():
            d = B.distribution
            total = sum(d.values())
            prob_dist = {k: (v / total) for k, v in d.items()}

            # Generate a run based on the probability distribution
            runs = list(prob_dist.keys())
            probabilities = list(prob_dist.values())
            iobs.append(iob)
            run_history.append(runs[np.random.choice(range(len(probabilities)), p=probabilities)])

        innings, overs, ballnumber = list(zip(*iobs))
        
        # Create a DataFrame to store the simulation results
        simul = pd.DataFrame({"innings": innings, "overs": overs, "ballnumber": ballnumber, "totalrun": run_history})
        
        # Identify wicket deliveries and replace -1 with 0 in the DataFrame
        simul['isWicketDelivery'] = (simul['totalrun'] == -1)
        simul.replace(-1, 0, inplace=True)

        # TODO: Crop the second inning after the game is finished (run is chased or all-out)
        # TODO: Insert player ability, currently same stats for each player

        return simul

simulator = Match()


# Define an API endpoint to get the simulated data
@app.route('/simulate_match', methods=['GET'])
def simulate_match():
    simulated_data = simulator.random_simulate().to_dict(orient='records')
    return jsonify(simulated_data)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)