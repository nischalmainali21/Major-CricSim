# CricSim

A cricket match analysis and simulation project, aiming to provide data analysis, visualization and simulation of matches. It uses the dataset of IPL matches from 2008-2022, uses two models, heuristic and LSTM, to develop the prediction for the matches.

This repo contains the backend, models, and notebooks for test builds and simulation of model. The fronted for the project is at [CricSim](https://github.com/nischalmainali21/Major-CricSim-Frontend).

The [deployed version](https://cricsim.vercel.app/) only works for for comparing players and does not have simulation capabilites. For the full project, clone this repo and the frontend.

# Installation

Create a virtual environment. Activate the environment.

```
clone https://github.com/nischalmainali21/Major-CricSim.git
cd Major-CricSim
pip install -r requirements.txt
```

Alos install `torch` as per the requirement of your hardware.

# Extra Files Required

Download and place the following files in appropriate location. All the extra files are available at [download](https://mega.nz/folder/mfxRDAoA#uyB5hBzuDbCqDs0wJqGDcg)

- final_stats_data.csv at Major-CricSim/derived and Major-CricSim/backend/cric_site/static_folder

- cleaned_stats_data.csv at Major-CricSim/backend/cric_site/static_folder/

- complete_ball_by_ball.csv at Major-CricSim/backend/cric_site/static_folder/all_data

# Usage

```
cd Major-CricSim/backend/cric_site
python manage.py runserver
```
