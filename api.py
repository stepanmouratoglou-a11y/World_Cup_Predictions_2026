from pydantic import BaseModel
from fastapi import FastAPI
from http.client import HTTPException
import joblib
import pandas as pd


app=FastAPI(title='World Cup Predictions')

rf_model=joblib.load("src/Models/Dumped_models/RandomForest_Classifier.pkl")
xgboost_model=joblib.load("src/Models/Dumped_models/XGBoost_Classifier.pkl")

CT=joblib.load("src/Models/Dumped_models/ColumnTransformer.pkl")
imputer=joblib.load("src/Models/Dumped_models/SimpleImputer.pkl")
team_data=joblib.load("src/Models/Dumped_models/TeamData.pkl")


world_cup_teams={
    'Group A':[
        'Mexico','South Africa','South Korea','Czechia'
    ],
    'Group B':[
        'Canada','Bosnia & Herzegovina','Qatar','Switzerland'
    ],
    'Group C':[
        'Brazil','Morocco','Haiti','Scotland'
    ],
    'Group D':[
        'USA','Paraguay','Australia','Türkiye'
    ],
    'Group E':[
        'Germany','Curaçao','Côte d\'Ivoire','Ecuador'
    ],
    'Group F':[
        'Netherlands','Japan','Sweden','Tunisia'
    ],
    'Group G':[
        'Belgium','Egypt','Iran','New Zealand'
    ],
    'Group H':[
        'Spain','Cape Verde','Saudi Arabia','Uruguay'
    ],
    'Group I':[
        'France','Senegal','Iraq','Norway'
    ],
    'Group J':[
        'Argentina','Algeria','Austria','Jordan'
    ],
    'Group K':[
        'Portugal','DR Congo','Uzbekistan','Colombia'
    ],
    'Group L':[
        'England','Croatia','Ghana','Panama'
    ]
}


class MatchFeatures(BaseModel):
    group:str
    home_team:str
    away_team:str
@app.post("/predict")
def make_prediction(match:MatchFeatures):
    if match.home_team not in world_cup_teams[match.group] and match.away_team not in world_cup_teams[match.group]:
        raise HTTPException(status_code=404,
                            detail=f"{match.home_team.capitalize()} and {match.away_team.capitalize()} are not in the 2026 WC")
    elif match.home_team not in world_cup_teams[match.group]:
        raise HTTPException(
            status_code=404,
            detail=f"{match.home_team} is not in the 2026 WC"
        )
    elif match.away_team not in world_cup_teams[match.group]:
        raise HTTPException(
            status_code=404,
            detail=f"{match.home_team} is not in the 2026 WC"
        )
    
    home=team_data[match.home_team]
    away=team_data[match.away_team]

    input_data={
        'tournament_name': 'World Cup',
        'Home_Big_Chances_5': home['Big Chances EMA 5'],
        'Home_Shots_5': home['Total Shots EMA 5'],
        'Home_Shots_on_Target_5': home['Shots on Target EMA 5'],
        'Home_touches_in_penalty_area_5': home['Touches in Penalty Area EMA 5'],
        'Home_big_chances_last_5': home['Big Chances Scored last 5'],
        'Home_final_third_entries_last_5': home['Final Third Entries last 5'],
        'Home_touches_in_penalty_area_last_5': home['Touches in Penalty Area last 5'],
        'Home_wins_last_5': home['Wins_Last_5'],
        'Home_goals_last_5': home['Goals_Last_5'],
        'Home_conceded_last_5': home['Conceded_Last_5'],
        'Home_gd_last_5': home['GD_Last_5'],
        
        'Away_Big_Chances_5': away['Big Chances EMA 5'],
        'Away_Shots_5': away['Total Shots EMA 5'],
        'Away_Shots_on_Target_5': away['Shots on Target EMA 5'],
        'Away_touches_in_penalty_area_5': away['Touches in Penalty Area EMA 5'],
        'Away_big_chances_last_5': away['Big Chances Scored last 5'],
        'Away_final_third_entries_last_5': away['Final Third Entries last 5'],
        'Away_touches_in_penalty_area_last_5': away['Touches in Penalty Area last 5'],
        'Away_wins_last_5': away['Wins_Last_5'],
        'Away_goals_last_5': away['Goals_Last_5'],
        'Away_conceded_last_5': away['Conceded_Last_5'],
        'Away_gd_last_5': away['GD_Last_5'],
        
        'Home_elo': home['Team ELO'],
        'Away_elo': away['Team ELO'],
        'ELO_diff': home['Team ELO'] - away['Team ELO']
    }

    input_data=pd.DataFrame(input_data)

    X=CT.transform(input_data)
    X=imputer.transform(X)

    xgb_probabilities=xgboost_model.predict_proba(X)[0]
    rf_probabilities=rf_model.predict_proba(X)[0]

    classes = ['Home Win', 'Draw','Away Win' ]

    return {
        "Match": f"{match.home_team} vs {match.away_team}",
        "Predictions": {
            "XGBoost": {classes[i]: float(round(xgb_probabilities[i]*100, 2)) for i in range(3)},
            "Random_Forest": {classes[i]: float(round(rf_probabilities[i]*100, 2)) for i in range(3)}
        }
    }
