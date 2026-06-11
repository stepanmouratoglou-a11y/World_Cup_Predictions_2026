import pandas as pd
import numpy as np
import sys
import os

def edit_tournament_name(raw_name, tournaments):
    raw_name_str = str(raw_name).strip()
    for keyword, macro_category in tournaments.items():
        if keyword in raw_name_str:
            return macro_category
    return "Friendlies & Minor"


def preprocessing(data):
    columns_to_keep = ['match_date', 'tournament_name', 'is_neutral_ground', 'Home Team',
                       'Away Team', 'home_final_score', 'away_final_score', 'home_big_chances',
                       'away_big_chances', 'home_total_shots', 'away_total_shots', 'home_shots_on_target',
                       'away_shots_on_target', 'home_big_chances_scored', 'away_big_chances_scored',
                       'home_final_third_entries', 'away_final_third_entries', 'home_touches_in_penalty_area',
                       'away_touches_in_penalty_area']

    data = data[columns_to_keep].copy()
    data = data.dropna(subset=['home_final_score', 'away_final_score', 'Home Team', 'Away Team'])

    tournaments = {
        "FIFA World Cup": "World Cup", "Finalissima": "World Cup",
        "Qual": "Qualifiers", "Qualification": "Qualifiers",
        "Nations League": "Nations League", "Euro,": "Major Continental",
        "Copa América": "Major Continental", "Africa Cup of Nations": "Major Continental",
        "AFC Asian Cup": "Major Continental", "Friendly": "Friendlies & Minor",
        "FIFA Series": "Friendlies & Minor", "Championship": "Friendlies & Minor",
        "Cup": "Friendlies & Minor"
    }

    data['home_final_score'] = pd.to_numeric(data['home_final_score'], errors='coerce')
    data['away_final_score'] = pd.to_numeric(data['away_final_score'], errors='coerce')

    data['tournament_name'] = data['tournament_name'].apply(edit_tournament_name, args=(tournaments,))
    data['match_date'] = pd.to_datetime(data['match_date'], errors='coerce').dt.normalize()

    data['Final Result'] = 'D'
    data.loc[data['home_final_score'] > data['away_final_score'], 'Final Result'] = 'H'
    data.loc[data['home_final_score'] < data['away_final_score'], 'Final Result'] = 'A'

    home_team_data = pd.DataFrame({
        'Team': data['Home Team'], 'Date': data['match_date'],
        'Big Chances': data['home_big_chances'], 'Total Shots': data['home_total_shots'],
        'Shots on Target': data['home_shots_on_target'], 'Big Chances Scored': data['home_big_chances_scored'],
        'Final Third Entries': data['home_final_third_entries'],
        'Touches in Penalty Area': data['home_touches_in_penalty_area'],
        'Final Result': data['Final Result'],
        'Goals Scored':data['home_final_score'],
        'Goals Conceded':data['away_final_score']
    })
    home_team_data['Wins'] = (data['Final Result'] == 'H').astype(int)
    home_team_data['Draws'] = (data['Final Result'] == 'D').astype(int)

    away_team_data = pd.DataFrame({
        'Team': data['Away Team'], 'Date': data['match_date'],
        'Big Chances': data['away_big_chances'], 'Total Shots': data['away_total_shots'],
        'Shots on Target': data['away_shots_on_target'], 'Big Chances Scored': data['away_big_chances_scored'],
        'Final Third Entries': data['away_final_third_entries'],
        'Touches in Penalty Area': data['away_touches_in_penalty_area'],
        'Final Result': data['Final Result'],
        'Goals Scored':data['away_final_score'],
        'Goals Conceded':data['home_final_score']
    })

    away_team_data['Wins'] = (data['Final Result'] == 'A').astype(int)
    away_team_data['Draws'] = (data['Final Result'] == 'D').astype(int)

    def edit_features(df, feature, span, min_periods):

        if feature not in ['Big Chances Scored','Final Third Entries']:
            df[f'{feature} EMA {span}'] = df.groupby('Team')[feature].transform(
                lambda x: x.shift(1).rolling(window=span, min_periods=min_periods).mean())
        if feature in ['Big Chances Scored','Final Third Entries','Touches in Penalty Area']:
            df[f'{feature} last {span}'] = df.groupby('Team')[feature].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=min_periods).sum())
        return df


    stats = pd.concat([home_team_data, away_team_data]).sort_values(['Team', 'Date']).reset_index(drop=True)
    stats['Wins_Last_5'] = stats.groupby('Team')['Wins'].transform(lambda x: x.shift(1).rolling(window=5).sum())
    stats['Goals_Last_5'] = stats.groupby('Team')['Goals Scored'].transform(lambda x:x.shift(1).rolling(window=5).sum())
    stats['Conceded_Last_5']=stats.groupby('Team')['Goals Conceded'].transform(lambda x:x.shift(1).rolling(window=5).sum())
    columns_to_transform = ['Big Chances', 'Total Shots', 'Shots on Target', 'Big Chances Scored',
                            'Final Third Entries', 'Touches in Penalty Area']

    stats['GD_Last_5']=stats['Goals_Last_5'] - stats['Conceded_Last_5']

    for col in columns_to_transform:
        if col not in ['Big Chances Scored', 'Final Third Entries']:
            stats = edit_features(stats, col, 5, 3)
        if col in ['Big Chances Scored', 'Final Third Entries','Touches in Penalty Area']:
            stats = edit_features(stats, col, 5, 1)

    stats=stats.drop(columns=['Goals Scored','Goals Conceded'])

    new_features = ([col for col in stats.columns if 'EMA' in col] + [col for col in stats.columns if ' last ' in col] +
                    ['Wins_Last_5','Goals_Last_5','Conceded_Last_5','GD_Last_5', 'Date', 'Team'])

    stats_to_merge = stats[new_features].drop_duplicates(subset=['Date', 'Team'])

    data = data.merge(stats_to_merge, left_on=['match_date', 'Home Team'], right_on=['Date', 'Team'],
                      how='left').rename(columns={
        'Big Chances EMA 5': 'Home_Big_Chances_5',
        'Total Shots EMA 5': 'Home_Shots_5',
        'Shots on Target EMA 5': 'Home_Shots_on_Target_5',
        'Touches in Penalty Area EMA 5': 'Home_touches_in_penalty_area_5',
        'Wins_Last_5': 'Home_wins_last_5',
        'Goals_Last_5':'Home_goals_last_5',
        'Conceded_Last_5':'Home_conceded_last_5',
        'GD_Last_5':'Home_gd_last_5',
        'Big Chances Scored last 5':'Home_big_chances_last_5',
        'Final Third Entries last 5':'Home_final_third_entries_last_5',
        'Touches in Penalty Area last 5':'Home_touches_in_penalty_area_last_5'
    }).drop(columns=['Team', 'Date'])

    data = data.merge(stats_to_merge, left_on=['match_date', 'Away Team'], right_on=['Date', 'Team'],
                      how='left').rename(columns={
        'Big Chances EMA 5': 'Away_Big_Chances_5',
        'Total Shots EMA 5': 'Away_Shots_5',
        'Shots on Target EMA 5': 'Away_Shots_on_Target_5',
        'Final Third Entries EMA 3': 'Away_final_third_entries_5',
        'Touches in Penalty Area EMA 5': 'Away_touches_in_penalty_area_5',
        'Wins_Last_5': 'Away_wins_last_5',
        'Goals_Last_5':'Away_goals_last_5',
        'Conceded_Last_5': 'Away_conceded_last_5',
        'GD_Last_5': 'Away_gd_last_5',
        'Big Chances Scored last 5':'Away_big_chances_last_5',
        'Final Third Entries last 5': 'Away_final_third_entries_last_5',
        'Touches in Penalty Area last 5': 'Away_touches_in_penalty_area_last_5'
        }).drop(columns=['Team', 'Date'])

    data, team_elos = calculate_team_elo(data)
    data['ELO_diff'] = data['Home_elo'] - data['Away_elo']

    raw_stats_to_drop = [
        'home_big_chances', 'away_big_chances', 'home_total_shots', 'away_total_shots',
        'home_shots_on_target', 'away_shots_on_target', 'home_big_chances_scored', 'away_big_chances_scored',
        'home_final_third_entries', 'away_final_third_entries', 'home_touches_in_penalty_area',
        'away_touches_in_penalty_area', 'home_goals_prevented', 'away_goals_prevented',
        'home_final_score', 'away_final_score', 'is_neutral_ground', 'match_date'
    ]

    data = data.drop(columns=raw_stats_to_drop, errors='ignore')
    def drop_random_nulls(data,column_name,drop:int):
        column = column_name
        null_rows = data[data[column].isnull()]
        rows_to_drop = null_rows.sample(n=min(drop, len(null_rows)), random_state=42)
        data = data.drop(rows_to_drop.index)
        data = data.reset_index(drop=True)
        return data

    data=drop_random_nulls(data,'Away_touches_in_penalty_area_5',250)
    data=drop_random_nulls(data,'Home_touches_in_penalty_area_5',250)


    #dir = '../../../Data/cleaned/'
    #file_name='training_data.csv'
    #os.makedirs(dir,exist_ok=True)
    #path=os.path.join(dir,file_name)
    #data = data[data.isnull().sum(axis=1) <= 8].reset_index(drop=True)
    #data.to_csv(path,index=False)
    return data,stats, team_elos


def calculate_team_elo(dataset):
    default_elo = 1500
    fifa_elos = {
        # Group A
        "Mexico": 1687.48,
        "South Africa": 1428.00,
        "South Korea": 1591.63,
        "Czechia": 1505.74,
        # Group B
        "Canada": 1559.48,
        "Bosnia": 1387.00,
        "Bosnia & Herzegovina": 1387.00,
        "Bosnia and Herzegovina": 1387.00,
        "Qatar": 1450.00,
        "Switzerland": 1650.07,
        # Group C
        "Brazil": 1761.16,
        "Morocco": 1755.87,
        "Haiti": 1293.00,
        "Scotland": 1505.00,
        # Group D
        "USA": 1673.13,
        "Paraguay": 1505.00,
        "Australia": 1579.34,
        "Türkiye": 1605.73,
        # Group E
        "Germany": 1690.37,
        "Curacao": 1294.00,
        "Curaçao": 1294.00,
        "Côte d'Ivoire": 1549.00,
        "Ecuador": 1605.00,
        # Group F
        "Netherlands": 1757.87,
        "Japan": 1660.43,
        "Sweden": 1516.00,
        "Tunisia": 1481.00,
        # Group G
        "Belgium": 1734.71,
        "Egypt": 1579.00,
        "Iran": 1619.58,
        "New Zealand": 1275.00,
        # Group H
        "Spain": 1876.40,
        "Cabo Verde": 1369.00,
        "Saudi Arabia": 1428.00,
        "Uruguay": 1673.07,
        # Group I
        "France": 1877.32,
        "Senegal": 1688.99,
        "Iraq": 1450.00,
        "Norway": 1559.00,
        # Group J
        "Argentina": 1874.81,
        "Algeria": 1585.00,
        "Austria": 1598.00,
        "Jordan": 1423.00,
        # Group K
        "Portugal": 1763.83,
        "DR Congo": 1476.00,
        "Uzbekistan": 1469.00,
        "Colombia": 1693.09,
        # Group L
        "England": 1825.97,
        "Croatia": 1717.07,
        "Ghana": 1370.00,
        "Panama": 1540.00
    }
    home_elo = []
    away_elo = []
    matches_played = {}
    teams_elo = {}

    for index, row in dataset.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        result = row['Final Result']

        if home_team not in teams_elo:
            teams_elo[home_team] = fifa_elos.get(home_team, default_elo)
            matches_played[home_team] = 0

        if away_team not in teams_elo:
            teams_elo[away_team] = fifa_elos.get(away_team, default_elo)
            matches_played[away_team] = 0

        current_home_elo = teams_elo[home_team]
        current_away_elo = teams_elo[away_team]

        expected_home_elo = expected_probability(current_home_elo, current_away_elo)
        expected_away_elo = expected_probability(current_away_elo, current_home_elo)

        if result == 'H':
            home_result, away_result = 1.0, 0.0
        elif result == 'D':
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0.0, 1.0

        home_k = k_Calculation(current_home_elo, matches_played[home_team])
        away_k = k_Calculation(current_away_elo, matches_played[away_team])

        teams_elo[home_team] = elo_rating(current_home_elo, expected_home_elo, home_result, home_k)
        teams_elo[away_team] = elo_rating(current_away_elo, expected_away_elo, away_result, away_k)

        matches_played[home_team] += 1
        matches_played[away_team] += 1

        home_elo.append(current_home_elo)
        away_elo.append(current_away_elo)

    dataset['Home_elo'] = home_elo
    dataset['Away_elo'] = away_elo
    return dataset, teams_elo


def expected_probability(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def elo_rating(elo_a, expected_probability, result, K=20):
    return elo_a + K * (result - expected_probability)


def k_Calculation(team_elo, matches_played):
    if team_elo > 1800:
        return 10
    elif matches_played > 20:
        return 15
    else:
        return 20