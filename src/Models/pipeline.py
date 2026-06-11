import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.ETL.Transform.feature_engineering import preprocessing
import joblib
from train import random_forest_classifier,logistic_regression,xgboost_model,artificial_neural_network
sys.path.remove(project_root)


print("Starting Process...")
messy_dataset=pd.read_csv('../../Data/raw/nations_data.csv')
dataset, team_stats, team_elos =  preprocessing(messy_dataset)
X=dataset.drop(columns=['Home Team','Away Team','Final Result'])
pd.DataFrame(X).to_csv('../../Data/cleaned/input_data.csv',index=False)
dataset.to_csv('../../Data/cleaned/training_data.csv',index=False)
y=dataset['Final Result']


labels={'A':0,'D':1,'H':2}
y=y.map(labels)

categorical_features=['tournament_name']
numerical_features=[
    'Home_Big_Chances_5',
    'Home_Shots_5',
    'Home_Shots_on_Target_5',
    'Home_touches_in_penalty_area_5',
    'Home_big_chances_last_5',
    'Home_final_third_entries_last_5',
    'Home_touches_in_penalty_area_last_5',
    'Home_wins_last_5',
    'Home_goals_last_5',
    'Home_conceded_last_5',
    'Home_gd_last_5',
    'Away_Big_Chances_5',
    'Away_Shots_5',
    'Away_Shots_on_Target_5',
    'Away_touches_in_penalty_area_5',
    'Away_big_chances_last_5',
    'Away_final_third_entries_last_5',
    'Away_touches_in_penalty_area_last_5',
    'Away_wins_last_5',
    'Away_goals_last_5',
    'Away_conceded_last_5',
    'Away_gd_last_5',
    'Home_elo',
    'Away_elo',
    'ELO_diff'
    ]

dir='../Models/Dumped_models'

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

print("started Preprocessing phase...")
CT=ColumnTransformer(transformers=[('encoder',OneHotEncoder(handle_unknown='ignore'),categorical_features)],
                     remainder='passthrough')
X_train=CT.fit_transform(X_train)
X_test=CT.transform(X_test)


print('Column Transformer Successful')
joblib.dump(CT,f"{dir}/ColumnTransformer.pkl")
print('Column Transformer was saved at: ')

imputer=SimpleImputer(strategy='median')
X_train=imputer.fit_transform(X_train)
X_test=imputer.transform(X_test)
print('Imputer Successful')
joblib.dump(imputer,f"{dir}/SimpleImputer.pkl")

sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)
print("Scaling Successful")
joblib.dump(sc,f'{dir}/StandardScaler.pkl')

print("Starting Training Process...")
print("======================================================")

xgboost=xgboost_model(X_train,y_train)
print("XGBoost Model Created Successfully")

random_forest=random_forest_classifier(X_train,y_train)
print("Random Forest Model Created Successfully")

regressor=logistic_regression(X_train_scaled,y_train)
print("Logistic Regression Model Created Successfully")

neural_network=artificial_neural_network(X_train_scaled,y_train)
print("Neural Network Created Successfully")

dir='../Models/Dumped_models'
os.makedirs(f'{dir}',exist_ok=True)

joblib.dump(xgboost,f'{dir}/XGBoost_Classifier.pkl')
print(f"Saved XGBoost Model to {dir}/XGBoost_Classifier.pkl")

joblib.dump(random_forest,f'{dir}/RandomForest_Classifier.pkl')
print(f"Saved Random Forest Model to {dir}/RandomForest_Classifier.pkl")

joblib.dump(regressor,f'{dir}/LogisticRegressor.pkl')
print(f"Saved Logistic Regression Model to {dir}/LogisticRegressor_Classifier.pkl")

joblib.dump(neural_network,f'{dir}/Neural_Network.pkl')
print(f"Saved Neural Network to {dir}/Neural_Network.pkl")

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

team_data={}
for group in world_cup_teams.keys():
    for team in world_cup_teams[group]:
        team_data[team]={
            'Team':team,
            'Team ELO':team_elos.get(team),
            'Tournament_Name':'World Cup',
            'Wins_Last_5':team_stats['Wins_Last_5'],
            'Goals_Last_5':team_stats['Goals_Last_5'],
            'Conceded_Last_5':team_stats['Conceded_Last_5'],
            'GD_Last_5':team_stats['GD_Last_5'],
            'Big Chances EMA 5':team_stats['Big Chances EMA 5'],
            'Total Shots EMA 5':team_stats['Total Shots EMA 5'],
            'Shots on Target EMA 5':team_stats['Shots on Target EMA 5'],
            'Big Chances Scored last 5':team_stats['Big Chances Scored last 5'],
            'Final Third Entries last 5':team_stats['Final Third Entries last 5'],
            'Touches in Penalty Area EMA 5':team_stats['Touches in Penalty Area EMA 5'],
            'Touches in Penalty Area last 5':team_stats['Touches in Penalty Area last 5']
        }

joblib.dump(team_data,f"{dir}/TeamData.pkl")
print(f"Saved Team Data at: {dir}/TeamData.pkl")

