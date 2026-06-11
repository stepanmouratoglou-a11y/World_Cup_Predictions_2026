import os
import sys
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import textwrap
from datetime import datetime

from PIL import Image

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
try:
    trophy_image_path = os.path.join(project_root, "Images", "World_Cup_Trophy", "World_Cup.png")
    favicon_img = Image.open(trophy_image_path)
except Exception:
    favicon_img = "⚽"

st.set_page_config(
    page_title="World Cup 2026 predictions made by Stepan",
    page_icon=favicon_img,
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "Main"


from src.ETL.Transform.feature_engineering import preprocessing


normalization_map = {
    "Bosnia": "Bosnia & Herzegovina",
    "Turkiye": "Türkiye",
    "Curacao": "Curaçao",
    "Cote Divoire": "Côte d'Ivoire",
    "Cote divoire": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde"
}

import base64

@st.cache_data
def get_base64_image(image_path):
    """Loads a local image and converts it to a base64 string for direct HTML rendering."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

flag_file_map = {
    "Mexico": "Mexico.png",
    "South Africa": "south africa.jpg",
    "South Korea": "south korea.png",
    "Czechia": "czechia.png",
    "Canada": "Canada.png",
    "Bosnia": "Bosnia.png",
    "USA": "USA.png",
    "Paraguay": "Paraguay.png",
    "Qatar": "qatar.png",
    "Switzerland": "Switzerland.png",
    "Brazil": "Brazil.png",
    "Morocco": "Morocco.png",
    "Haiti": "haiti.png",
    "Scotland": "Scotland.png",
    "Australia": "Australia.png",
    "Turkiye": "Turkey.png",
    "Germany": "Germany.png",
    "Curacao": "Curacao.png",
    "Netherlands": "Netherlands.png",
    "Japan": "Japan.webp",
    "Cote Divoire": "Cote divoire.png",
    "Cote divoire": "Cote divoire.png",
    "Ecuador": "Ecuador.png",
    "Sweden": "Sweden.png",
    "Tunisia": "Tunisia.png",
    "Spain": "Spain.png",
    "Cape Verde": "Cape Verde.png",
    "Belgium": "Belgium.png",
    "Egypt": "Egypt.png",
    "Saudi Arabia": "Saudi Arabia.png",
    "Uruguay": "Uruguay.png",
    "Iran": "Iran.png",
    "New Zealand": "New Zealand.png",
    "France": "France.png",
    "Senegal": "Senegal.png",
    "Iraq": "Iraq.png",
    "Norway": "Norway.png",
    "Argentina": "Argentina.png",
    "Algeria": "Algeria.png",
    "Austria": "Austria.png",
    "Jordan": "Jordan.png",
    "Portugal": "Portugal.png",
    "DR Congo": "Congo.png",
    "England": "England.png",
    "Croatia": "Croatia.png",
    "Ghana": "Ghana.png",
    "Panama": "Panama.png",
    "Uzbekistan": "Uzbekistan.png",
    "Colombia": "Colombia.png"
}

trophy_path = os.path.join(project_root, "Images", "World_Cup_Trophy", "World_Cup.png")
trophy_base64 = get_base64_image(trophy_path)

def get_models_mtime():
    """Returns a tuple of modification times for all serialized model files to trigger auto-reload."""
    models_dir = os.path.join(project_root, "src", "Models", "Dumped_models")
    files = [
        "ColumnTransformer.pkl",
        "SimpleImputer.pkl",
        "StandardScaler.pkl",
        "RandomForest_Classifier.pkl",
        "XGBoost_Classifier.pkl",
        "Neural_Network.pkl"
    ]
    mtimes = []
    for f in files:
        path = os.path.join(models_dir, f)
        if os.path.exists(path):
            mtimes.append(os.path.getmtime(path))
        else:
            mtimes.append(0.0)
    return tuple(mtimes)


@st.cache_resource
def load_models_and_preprocessors(mtimes):
    """Loads classification models and pipelines from Dumped_models."""
    models_dir = os.path.join(project_root, "src", "Models", "Dumped_models")
    ct = joblib.load(os.path.join(models_dir, "ColumnTransformer.pkl"))
    imputer = joblib.load(os.path.join(models_dir, "SimpleImputer.pkl"))
    sc = joblib.load(os.path.join(models_dir, "StandardScaler.pkl"))
    rf_model = joblib.load(os.path.join(models_dir, "RandomForest_Classifier.pkl"))
    xgb_model = joblib.load(os.path.join(models_dir, "XGBoost_Classifier.pkl"))
    ann_model = joblib.load(os.path.join(models_dir, "Neural_Network.pkl"))
    return ct, imputer, sc, rf_model, xgb_model, ann_model


@st.cache_data
def get_latest_stats(raw_path):
    """Loads raw dataset, runs feature engineering pipeline to get latest ELO and stats."""
    messy_dataset = pd.read_csv(raw_path)
    dataset, stats, team_elos = preprocessing(messy_dataset)
    
    median_stats = stats.median(numeric_only=True)
    return stats, team_elos, median_stats

mtimes = get_models_mtime()
ct, imputer, sc, rf_model, xgb_model, ann_model = load_models_and_preprocessors(mtimes)
raw_data_path = os.path.join(project_root, "Data", "raw", "nations_data.csv")
stats, team_elos, median_stats = get_latest_stats(raw_data_path)


# Helper Prediction Functions
def get_team_features_row(team_name):
    """Retrieves the latest rolling stats and features row for a specific team."""
    matching = stats[stats['Team'] == team_name]
    if len(matching) > 0:
        return matching.iloc[-1]
    return median_stats


def get_match_features_df(home_team, away_team):
    """Constructs the feature DataFrame for the prediction pipelines."""
    home_row = get_team_features_row(home_team)
    away_row = get_team_features_row(away_team)
    
    h_elo = team_elos.get(home_team, 1500)
    a_elo = team_elos.get(away_team, 1500)
    
    data = {
        'tournament_name': ['World Cup'],
        'Home_Big_Chances_5': [home_row['Big Chances EMA 5']],
        'Home_Shots_5': [home_row['Total Shots EMA 5']],
        'Home_Shots_on_Target_5': [home_row['Shots on Target EMA 5']],
        'Home_touches_in_penalty_area_5': [home_row['Touches in Penalty Area EMA 5']],
        'Home_big_chances_last_5': [home_row['Big Chances Scored last 5']],
        'Home_final_third_entries_last_5': [home_row['Final Third Entries last 5']],
        'Home_touches_in_penalty_area_last_5': [home_row['Touches in Penalty Area last 5']],
        'Home_wins_last_5': [home_row['Wins_Last_5']],
        'Home_goals_last_5': [home_row['Goals_Last_5']],
        'Home_conceded_last_5': [home_row['Conceded_Last_5']],
        'Home_gd_last_5': [home_row['GD_Last_5']],
        'Away_Big_Chances_5': [away_row['Big Chances EMA 5']],
        'Away_Shots_5': [away_row['Total Shots EMA 5']],
        'Away_Shots_on_Target_5': [away_row['Shots on Target EMA 5']],
        'Away_touches_in_penalty_area_5': [away_row['Touches in Penalty Area EMA 5']],
        'Away_big_chances_last_5': [away_row['Big Chances Scored last 5']],
        'Away_final_third_entries_last_5': [away_row['Final Third Entries last 5']],
        'Away_touches_in_penalty_area_last_5': [away_row['Touches in Penalty Area last 5']],
        'Away_wins_last_5': [away_row['Wins_Last_5']],
        'Away_goals_last_5': [away_row['Goals_Last_5']],
        'Away_conceded_last_5': [away_row['Conceded_Last_5']],
        'Away_gd_last_5': [away_row['GD_Last_5']],
        'Home_elo': [h_elo],
        'Away_elo': [a_elo],
        'ELO_diff': [h_elo - a_elo]
    }
    return pd.DataFrame(data)


@st.cache_data
def predict_match_probs(team1, team2):
    """Calculates averaged Home/Away probabilities for RF, XGBoost, and ANN."""
    h_norm = normalization_map.get(team1, team1)
    a_norm = normalization_map.get(team2, team2)
    
    fwd_df = get_match_features_df(h_norm, a_norm)
    fwd_trans = ct.transform(fwd_df)
    fwd_imp = imputer.transform(fwd_trans)
    fwd_scaled = sc.transform(fwd_imp)
    
    rf_probs_fwd = rf_model.predict_proba(fwd_imp)[0]
    xgb_probs_fwd = xgb_model.predict_proba(fwd_imp)[0]
    ann_probs_fwd = ann_model.predict_proba(fwd_scaled)[0]
    
    bwd_df = get_match_features_df(a_norm, h_norm)
    bwd_trans = ct.transform(bwd_df)
    bwd_imp = imputer.transform(bwd_trans)
    bwd_scaled = sc.transform(bwd_imp)
    
    rf_probs_bwd = rf_model.predict_proba(bwd_imp)[0]
    xgb_probs_bwd = xgb_model.predict_proba(bwd_imp)[0]
    ann_probs_bwd = ann_model.predict_proba(bwd_scaled)[0]
    
    # RF
    rf_t1_win = (rf_probs_fwd[2] + rf_probs_bwd[0]) / 2.0
    rf_draw   = (rf_probs_fwd[1] + rf_probs_bwd[1]) / 2.0
    rf_t2_win = (rf_probs_fwd[0] + rf_probs_bwd[2]) / 2.0
    
    # XGBoost
    xgb_t1_win = (xgb_probs_fwd[2] + xgb_probs_bwd[0]) / 2.0
    xgb_draw   = (xgb_probs_fwd[1] + xgb_probs_bwd[1]) / 2.0
    xgb_t2_win = (xgb_probs_fwd[0] + xgb_probs_bwd[2]) / 2.0
    
    # ANN
    ann_t1_win = (ann_probs_fwd[2] + ann_probs_bwd[0]) / 2.0
    ann_draw   = (ann_probs_fwd[1] + ann_probs_bwd[1]) / 2.0
    ann_t2_win = (ann_probs_fwd[0] + ann_probs_bwd[2]) / 2.0
    
    return {
        "rf": (rf_t1_win * 100, rf_draw * 100, rf_t2_win * 100),
        "xgb": (xgb_t1_win * 100, xgb_draw * 100, xgb_t2_win * 100),
        "ann": (ann_t1_win * 100, ann_draw * 100, ann_t2_win * 100)
    }


def find_third_place_matching(qualified_groups):
    matches = [
        {"id": 74, "allowed": ['A', 'B', 'C', 'D', 'F']},
        {"id": 77, "allowed": ['C', 'D', 'F', 'G', 'H']},
        {"id": 79, "allowed": ['C', 'E', 'F', 'H', 'I']},
        {"id": 80, "allowed": ['E', 'H', 'I', 'J', 'K']},
        {"id": 81, "allowed": ['B', 'E', 'F', 'I', 'J']},
        {"id": 82, "allowed": ['A', 'E', 'H', 'I', 'J']},
        {"id": 85, "allowed": ['E', 'F', 'G', 'I', 'J']},
        {"id": 87, "allowed": ['D', 'E', 'I', 'J', 'L']}
    ]
    assignment = {}
    used = set()
    
    def backtrack(match_idx):
        if match_idx == len(matches):
            return True
        match = matches[match_idx]
        for g in sorted(qualified_groups):
            if g in match["allowed"] and g not in used:
                used.add(g)
                assignment[match["id"]] = g
                if backtrack(match_idx + 1):
                    return True
                used.remove(g)
                del assignment[match["id"]]
        return False
        
    if backtrack(0):
        return assignment
    else:
        remaining = sorted(list(qualified_groups))
        fallback = {}
        for m in matches:
            if remaining:
                fallback[m["id"]] = remaining.pop(0)
            else:
                fallback[m["id"]] = "A"
        return fallback


def simulate_knockout_match(team1, team2):
    probs = predict_match_probs(team1, team2)
    rf_t1, rf_draw, rf_t2 = probs["rf"]
    xgb_t1, xgb_draw, xgb_t2 = probs["xgb"]
    ann_t1, ann_draw, ann_t2 = probs["ann"]
    
    #Let's not play with draws
    rf_vote = team1 if rf_t1 >= rf_t2 else team2
    xgb_vote = team1 if xgb_t1 >= xgb_t2 else team2
    ann_vote = team1 if ann_t1 >= ann_t2 else team2
    
    votes = [rf_vote, xgb_vote, ann_vote]
    winner = team1 if votes.count(team1) >= 2 else team2
    loser = team2 if winner == team1 else team1
            
    return {
        "winner": winner,
        "loser": loser,
        "probs": probs
    }


@st.cache_data
def simulate_world_cup(matches_list):
    groups = {}
    for _, t1, t2, grp in matches_list:
        if grp not in groups:
            groups[grp] = {}
        if t1 not in groups[grp]:
            groups[grp][t1] = {"team": t1, "points": 0, "wins": 0, "draws": 0, "losses": 0}
        if t2 not in groups[grp]:
            groups[grp][t2] = {"team": t2, "points": 0, "wins": 0, "draws": 0, "losses": 0}
            
    for _, t1, t2, grp in matches_list:
        probs = predict_match_probs(t1, t2)
        rf_t1, rf_draw, rf_t2 = probs["rf"]
        xgb_t1, xgb_draw, xgb_t2 = probs["xgb"]
        ann_t1, ann_draw, ann_t2 = probs["ann"]
        
        rf_pred = 'H' if rf_t1 > rf_draw and rf_t1 > rf_t2 else ('A' if rf_t2 > rf_draw and rf_t2 > rf_t1 else 'D')
        xgb_pred = 'H' if xgb_t1 > xgb_draw and xgb_t1 > xgb_t2 else ('A' if xgb_t2 > xgb_draw and xgb_t2 > xgb_t1 else 'D')
        ann_pred = 'H' if ann_t1 > ann_draw and ann_t1 > ann_t2 else ('A' if ann_t2 > ann_draw and ann_t2 > ann_t1 else 'D')
        
        votes = [rf_pred, xgb_pred, ann_pred]
        counts = {c: votes.count(c) for c in ['H', 'D', 'A']}
        
        majority_result = None
        for c, count in counts.items():
            if count >= 2:
                majority_result = c
                break
        
        if majority_result is None:
            p1 = (rf_t1 + xgb_t1 + ann_t1) / 3.0
            pd = (rf_draw + xgb_draw + ann_draw) / 3.0
            p2 = (rf_t2 + xgb_t2 + ann_t2) / 3.0
            if p1 > pd and p1 > p2:
                majority_result = 'H'
            elif p2 > pd and p2 > p1:
                majority_result = 'A'
            else:
                majority_result = 'D'
                
        if majority_result == 'H':
            groups[grp][t1]["points"] += 3
            groups[grp][t1]["wins"] += 1
            groups[grp][t2]["losses"] += 1
        elif majority_result == 'A':
            groups[grp][t2]["points"] += 3
            groups[grp][t2]["wins"] += 1
            groups[grp][t1]["losses"] += 1
        else:
            groups[grp][t1]["points"] += 1
            groups[grp][t1]["draws"] += 1
            groups[grp][t2]["points"] += 1
            groups[grp][t2]["draws"] += 1
            
    ranked_groups = {}
    for grp, teams_dict in groups.items():
        teams_list = list(teams_dict.values())
        teams_list.sort(key=lambda x: (x["points"], team_elos.get(x["team"], 1500)), reverse=True)
        ranked_groups[grp] = teams_list
        
    third_places = []
    for grp, teams_list in ranked_groups.items():
        grp_letter = grp.replace("Group ", "")
        third_team = teams_list[2].copy()
        third_team["group"] = grp_letter
        third_places.append(third_team)
        
    third_places.sort(key=lambda x: (x["points"], team_elos.get(x["team"], 1500)), reverse=True)
    
    advancing_third_places = third_places[:8]
    advancing_third_groups = {item["group"] for item in advancing_third_places}
    
    third_place_mapping = find_third_place_matching(advancing_third_groups)
    
    r32_results = {}
    r32_fixtures = [
        {"id": 73, "date": "Sunday, 28 June 2026", "stadium": "Los Angeles Stadium", "t1": ranked_groups["Group A"][1]["team"], "t2": ranked_groups["Group B"][1]["team"]},
        {"id": 74, "date": "Monday, 29 June 2026", "stadium": "Boston Stadium", "t1": ranked_groups["Group E"][0]["team"], "t2": ranked_groups[f"Group {third_place_mapping[74]}"][2]["team"]},
        {"id": 75, "date": "Monday, 29 June 2026", "stadium": "Estadio Monterrey", "t1": ranked_groups["Group F"][0]["team"], "t2": ranked_groups["Group C"][1]["team"]},
        {"id": 76, "date": "Monday, 29 June 2026", "stadium": "Houston Stadium", "t1": ranked_groups["Group C"][0]["team"], "t2": ranked_groups["Group F"][1]["team"]},
        {"id": 77, "date": "Tuesday, 30 June 2026", "stadium": "New York New Jersey Stadium", "t1": ranked_groups["Group I"][0]["team"], "t2": ranked_groups[f"Group {third_place_mapping[77]}"][2]["team"]},
        {"id": 78, "date": "Tuesday, 30 June 2026", "stadium": "Dallas Stadium", "t1": ranked_groups["Group E"][1]["team"], "t2": ranked_groups["Group I"][1]["team"]},
        {"id": 79, "date": "Tuesday, 30 June 2026", "stadium": "Mexico City Stadium", "t1": ranked_groups["Group A"][0]["team"], "t2": ranked_groups[f"Group {third_place_mapping[79]}"][2]["team"]},
        {"id": 80, "date": "Wednesday, 1 July 2026", "stadium": "Atlanta Stadium", "t1": ranked_groups["Group L"][0]["team"], "t2": ranked_groups[f"Group {third_place_mapping[80]}"][2]["team"]},
        {"id": 81, "date": "Wednesday, 1 July 2026", "stadium": "San Francisco Bay Area Stadium", "t1": ranked_groups["Group D"][0]["team"], "t2": ranked_groups[f"Group {third_place_mapping[81]}"][2]["team"]},
        {"id": 82, "date": "Wednesday, 1 July 2026", "stadium": "Seattle Stadium", "t1": ranked_groups["Group G"][0]["team"], "t2": ranked_groups[f"Group {third_place_mapping[82]}"][2]["team"]},
        {"id": 83, "date": "Thursday, 2 July 2026", "stadium": "Toronto Stadium", "t1": ranked_groups["Group K"][1]["team"], "t2": ranked_groups["Group L"][1]["team"]},
        {"id": 84, "date": "Thursday, 2 July 2026", "stadium": "Los Angeles Stadium", "t1": ranked_groups["Group H"][0]["team"], "t2": ranked_groups["Group J"][1]["team"]},
        {"id": 85, "date": "Thursday, 2 July 2026", "stadium": "BC Place Vancouver", "t1": ranked_groups["Group B"][0]["team"], "t2": ranked_groups[f"Group {third_place_mapping[85]}"][2]["team"]},
        {"id": 86, "date": "Friday, 3 July 2026", "stadium": "Miami Stadium", "t1": ranked_groups["Group J"][0]["team"], "t2": ranked_groups["Group H"][1]["team"]},
        {"id": 87, "date": "Friday, 3 July 2026", "stadium": "Kansas City Stadium", "t1": ranked_groups["Group K"][0]["team"], "t2": ranked_groups[f"Group {third_place_mapping[87]}"][2]["team"]},
        {"id": 88, "date": "Friday, 3 July 2026", "stadium": "Dallas Stadium", "t1": ranked_groups["Group D"][1]["team"], "t2": ranked_groups["Group G"][1]["team"]}
    ]
    
    for fixture in r32_fixtures:
        res = simulate_knockout_match(fixture["t1"], fixture["t2"])
        r32_results[fixture["id"]] = {
            "id": fixture["id"],
            "date": fixture["date"],
            "stadium": fixture["stadium"],
            "t1": fixture["t1"],
            "t2": fixture["t2"],
            "winner": res["winner"],
            "loser": res["loser"],
            "probs": res["probs"]
        }
        
    # R16 Simulation
    r16_results = {}
    r16_fixtures = [
        {"id": 89, "date": "Saturday, 4 July 2026", "stadium": "Philadelphia Stadium", "t1_match": 74, "t2_match": 77},
        {"id": 90, "date": "Saturday, 4 July 2026", "stadium": "Houston Stadium", "t1_match": 73, "t2_match": 75},
        {"id": 91, "date": "Sunday, 5 July 2026", "stadium": "New York New Jersey Stadium", "t1_match": 76, "t2_match": 78},
        {"id": 92, "date": "Sunday, 5 July 2026", "stadium": "Mexico City Stadium", "t1_match": 79, "t2_match": 80},
        {"id": 93, "date": "Monday, 6 July 2026", "stadium": "Dallas Stadium", "t1_match": 83, "t2_match": 84},
        {"id": 94, "date": "Monday, 6 July 2026", "stadium": "Seattle Stadium", "t1_match": 81, "t2_match": 82},
        {"id": 95, "date": "Tuesday, 7 July 2026", "stadium": "Atlanta Stadium", "t1_match": 86, "t2_match": 88},
        {"id": 96, "date": "Tuesday, 7 July 2026", "stadium": "BC Place Vancouver", "t1_match": 85, "t2_match": 87}
    ]
    
    for fixture in r16_fixtures:
        t1 = r32_results[fixture["t1_match"]]["winner"]
        t2 = r32_results[fixture["t2_match"]]["winner"]
        res = simulate_knockout_match(t1, t2)
        r16_results[fixture["id"]] = {
            "id": fixture["id"],
            "date": fixture["date"],
            "stadium": fixture["stadium"],
            "t1": t1,
            "t2": t2,
            "winner": res["winner"],
            "loser": res["loser"],
            "probs": res["probs"]
        }
        
    # QF Simulation
    qf_results = {}
    qf_fixtures = [
        {"id": 97, "date": "Thursday, 9 July 2026", "stadium": "Boston Stadium", "t1_match": 89, "t2_match": 90},
        {"id": 98, "date": "Friday, 10 July 2026", "stadium": "Los Angeles Stadium", "t1_match": 93, "t2_match": 94},
        {"id": 99, "date": "Saturday, 11 July 2026", "stadium": "Miami Stadium", "t1_match": 91, "t2_match": 92},
        {"id": 100, "date": "Saturday, 11 July 2026", "stadium": "Kansas City Stadium", "t1_match": 95, "t2_match": 96}
    ]
    
    for fixture in qf_fixtures:
        t1 = r16_results[fixture["t1_match"]]["winner"]
        t2 = r16_results[fixture["t2_match"]]["winner"]
        res = simulate_knockout_match(t1, t2)
        qf_results[fixture["id"]] = {
            "id": fixture["id"],
            "date": fixture["date"],
            "stadium": fixture["stadium"],
            "t1": t1,
            "t2": t2,
            "winner": res["winner"],
            "loser": res["loser"],
            "probs": res["probs"]
        }
        
    sf_results = {}
    sf_fixtures = [
        {"id": 101, "date": "Tuesday, 14 July 2026", "stadium": "Dallas Stadium", "t1_match": 97, "t2_match": 98},
        {"id": 102, "date": "Wednesday, 15 July 2026", "stadium": "Atlanta Stadium", "t1_match": 99, "t2_match": 100}
    ]
    
    for fixture in sf_fixtures:
        t1 = qf_results[fixture["t1_match"]]["winner"]
        t2 = qf_results[fixture["t2_match"]]["winner"]
        res = simulate_knockout_match(t1, t2)
        sf_results[fixture["id"]] = {
            "id": fixture["id"],
            "date": fixture["date"],
            "stadium": fixture["stadium"],
            "t1": t1,
            "t2": t2,
            "winner": res["winner"],
            "loser": res["loser"],
            "probs": res["probs"]
        }
        
    t1_bronze = sf_results[101]["loser"]
    t2_bronze = sf_results[102]["loser"]
    res_bronze = simulate_knockout_match(t1_bronze, t2_bronze)
    bronze_result = {
        "id": 103,
        "date": "Saturday, 18 July 2026",
        "stadium": "Miami Stadium",
        "t1": t1_bronze,
        "t2": t2_bronze,
        "winner": res_bronze["winner"],
        "loser": res_bronze["loser"],
        "probs": res_bronze["probs"]
    }
    
    # Final Simulation
    t1_final = sf_results[101]["winner"]
    t2_final = sf_results[102]["winner"]
    res_final = simulate_knockout_match(t1_final, t2_final)
    final_result = {
        "id": 104,
        "date": "Sunday, 19 July 2026",
        "stadium": "New York New Jersey Stadium",
        "t1": t1_final,
        "t2": t2_final,
        "winner": res_final["winner"],
        "loser": res_final["loser"],
        "probs": res_final["probs"]
    }
    
    return {
        "ranked_groups": ranked_groups,
        "third_places": third_places,
        "advancing_third_places": advancing_third_places,
        "third_place_mapping": third_place_mapping,
        "r32_results": r32_results,
        "r16_results": r16_results,
        "qf_results": qf_results,
        "sf_results": sf_results,
        "bronze_result": bronze_result,
        "final_result": final_result
    }


def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.split("\n"))


def render_group_standings_table(grp_name, teams_list, advancing_third_groups):
    html = f"""
    <table class="standings-table">
        <thead>
            <tr>
                <th style="width: 12%; text-align: center;">Pos</th>
                <th style="width: 48%;">Team</th>
                <th style="width: 16%; text-align: center;">Pts</th>
                <th style="width: 8%; text-align: center;">W</th>
                <th style="width: 8%; text-align: center;">D</th>
                <th style="width: 8%; text-align: center;">L</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, t in enumerate(teams_list):
        pos = i + 1
        team_name = t["team"]
        pts = t["points"]
        w = t["wins"]
        d = t["draws"]
        l = t["losses"]
        
        tr_class = ""
        if pos == 1:
            tr_class = 'class="advancing-1"'
            pos_badge = '<span class="rank-badge rank-1">1</span>'
        elif pos == 2:
            tr_class = 'class="advancing-2"'
            pos_badge = '<span class="rank-badge rank-2">2</span>'
        elif pos == 3:
            grp_let = grp_name.replace("Group ", "")
            if grp_let in advancing_third_groups:
                tr_class = 'class="advancing-3"'
                pos_badge = '<span class="rank-badge rank-3">3</span>'
            else:
                tr_class = 'class="advancing-4"'
                pos_badge = '<span class="rank-badge rank-4">3</span>'
        else:
            tr_class = 'class="advancing-4"'
            pos_badge = '<span class="rank-badge rank-4">4</span>'
            
        flag_file = flag_file_map.get(team_name, "")
        flag_path = os.path.join(project_root, "Images", "Flags", flag_file)
        flag_b64 = get_base64_image(flag_path)
        ext = flag_file.split('.')[-1] if flag_file else "png"
        ext = "jpeg" if ext == "jpg" else ext
        
        flag_html = f'<img src="data:image/{ext};base64,{flag_b64}" style="height: 11px; width: auto; margin-left: 6px; border-radius: 2px; vertical-align: middle;"/>' if flag_b64 else ''
        
        html += f"""
            <tr {tr_class}>
                <td style="text-align: center;">{pos_badge}</td>
                <td><span style="font-weight: 600;">{team_name}</span> {flag_html}</td>
                <td style="text-align: center; font-weight: 700;">{pts}</td>
                <td style="text-align: center; color: #8fa0c0;">{w}</td>
                <td style="text-align: center; color: #8fa0c0;">{d}</td>
                <td style="text-align: center; color: #8fa0c0;">{l}</td>
            </tr>
        """
    html += "</tbody></table>"
    return clean_html(html)


def render_third_places_table(third_places_list, advancing_third_groups):
    html = """
    <table class="standings-table" style="max-width: 600px; margin: 0 auto;">
        <thead>
            <tr>
                <th style="width: 12%; text-align: center;">Pos</th>
                <th style="width: 18%; text-align: center;">Group</th>
                <th style="width: 46%;">Team</th>
                <th style="width: 12%; text-align: center;">Pts</th>
                <th style="width: 12%; text-align: center;">ELO</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, t in enumerate(third_places_list):
        pos = i + 1
        grp = t["group"]
        team_name = t["team"]
        pts = t["points"]
        elo = team_elos.get(team_name, 1500)
        
        is_advancing = grp in advancing_third_groups
        tr_class = 'class="advancing-1"' if is_advancing else 'class="advancing-4"'
        pos_badge = f'<span class="rank-badge rank-1">{pos}</span>' if is_advancing else f'<span class="rank-badge rank-4">{pos}</span>'
        
        flag_file = flag_file_map.get(team_name, "")
        flag_path = os.path.join(project_root, "Images", "Flags", flag_file)
        flag_b64 = get_base64_image(flag_path)
        ext = flag_file.split('.')[-1] if flag_file else "png"
        ext = "jpeg" if ext == "jpg" else ext
        
        flag_html = f'<img src="data:image/{ext};base64,{flag_b64}" style="height: 11px; width: auto; margin-left: 6px; border-radius: 2px; vertical-align: middle;"/>' if flag_b64 else ''
        
        html += f"""
            <tr {tr_class}>
                <td style="text-align: center;">{pos_badge}</td>
                <td style="text-align: center; font-weight: bold; color: #1f77b4;">{grp}</td>
                <td><span style="font-weight: 600;">{team_name}</span> {flag_html}</td>
                <td style="text-align: center; font-weight: 700;">{pts}</td>
                <td style="text-align: center; color: #8fa0c0;">{elo}</td>
            </tr>
        """
    html += "</tbody></table>"
    return clean_html(html)


def render_knockout_match_card(match_data):
    m_id = match_data["id"]
    date_str = match_data["date"]
    stadium = match_data["stadium"]
    t1 = match_data["t1"]
    t2 = match_data["t2"]
    winner = match_data["winner"]
    probs = match_data["probs"]
    
    rf_t1, rf_draw, rf_t2 = probs["rf"]
    xgb_t1, xgb_draw, xgb_t2 = probs["xgb"]
    ann_t1, ann_draw, ann_t2 = probs["ann"]
    
    t1_flag = flag_file_map.get(t1, "")
    t2_flag = flag_file_map.get(t2, "")
    
    t1_path = os.path.join(project_root, "Images", "Flags", t1_flag)
    t2_path = os.path.join(project_root, "Images", "Flags", t2_flag)
    
    t1_b64 = get_base64_image(t1_path)
    t2_b64 = get_base64_image(t2_path)
    
    ext1 = t1_flag.split('.')[-1] if t1_flag else "png"
    ext2 = t2_flag.split('.')[-1] if t2_flag else "png"
    ext1 = "jpeg" if ext1 == "jpg" else ext1
    ext2 = "jpeg" if ext2 == "jpg" else ext2
    
    t1_img = f'<img src="data:image/{ext1};base64,{t1_b64}" style="height: 13px; width: auto; vertical-align: middle; margin-left: 5px; border-radius: 2px;"/>' if t1_b64 else ''
    t2_img = f'<img src="data:image/{ext2};base64,{t2_b64}" style="height: 13px; width: auto; vertical-align: middle; margin-right: 5px; border-radius: 2px;"/>' if t2_b64 else ''
    
    t1_style = 'font-weight: 700; color: #2ecc71;' if winner == t1 else 'font-weight: 600; color: #ffffff;'
    t2_style = 'font-weight: 700; color: #2ecc71;' if winner == t2 else 'font-weight: 600; color: #ffffff;'
    
    winner_badge = f"""
    <div style="margin-top: 10px; text-align: center;">
        <span style="background-color: rgba(46, 204, 113, 0.12); color: #2ecc71; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;">
            👉 {winner} Advances
        </span>
    </div>
    """
    
    html = f"""
    <div class="match-card" style="margin-bottom: 20px; border: 1px solid #282f42; padding: 12px; background-color: #1a1e2a;">
        <div style="font-size: 0.7rem; color: #8fa0c0; font-weight: bold; margin-bottom: 4px; text-transform: uppercase; text-align: center;">
            Match {m_id} • {date_str}
        </div>
        <div style="font-size: 0.65rem; color: #5dade2; text-align: center; margin-bottom: 10px; font-weight: 600;">
            📍 {stadium}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="flex: 1; text-align: right; padding-right: 10px; {t1_style}">{t1} {t1_img}</div>
            <div class="vs-badge" style="font-size: 0.65rem; padding: 2px 6px;">VS</div>
            <div style="flex: 1; text-align: left; padding-left: 10px; {t2_style}">{t2_img} {t2}</div>
        </div>
        <div class="pred-section" style="border-top: 1px solid #232837; padding-top: 8px;">
            <div class="pred-row">
                <div class="pred-title rf-title" style="font-size: 0.68rem; margin-bottom: 2px;">🔵 Random Forest</div>
                <div class="pred-bars rf-bars" style="font-size: 0.68rem; padding: 3px 6px;">
                    <span>{t1}: {rf_t1:.1f}%</span>
                    <span>Draw: {rf_draw:.1f}%</span>
                    <span>{t2}: {rf_t2:.1f}%</span>
                </div>
            </div>
            <div class="pred-row" style="margin-top: 4px;">
                <div class="pred-title xgb-title" style="font-size: 0.68rem; margin-bottom: 2px;">🟢 XGBoost</div>
                <div class="pred-bars xgb-bars" style="font-size: 0.68rem; padding: 3px 6px;">
                    <span>{t1}: {xgb_t1:.1f}%</span>
                    <span>Draw: {xgb_draw:.1f}%</span>
                    <span>{t2}: {xgb_t2:.1f}%</span>
                </div>
            </div>
            <div class="pred-row" style="margin-top: 4px;">
                <div class="pred-title ann-title" style="font-size: 0.68rem; margin-bottom: 2px;">🟡 Neural Network</div>
                <div class="pred-bars ann-bars" style="font-size: 0.68rem; padding: 3px 6px;">
                    <span>{t1}: {ann_t1:.1f}%</span>
                    <span>Draw: {ann_draw:.1f}%</span>
                    <span>{t2}: {ann_t2:.1f}%</span>
                </div>
            </div>
        </div>
        {winner_badge}
    </div>
    """
    return clean_html(html)


st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0d0f14;
    }
    
    /* Navigation Buttons Style Override */
    div.stButton > button[kind="primary"] {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: bold !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 8px 20px !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 10px rgba(31, 119, 180, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1a6596 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 14px rgba(31, 119, 180, 0.4) !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background-color: #161a23 !important;
        color: #8fa0c0 !important;
        border: 1px solid #282f42 !important;
        font-weight: bold !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 8px 20px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #242b3d !important;
        color: #ffffff !important;
        border-color: #1f77b4 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Compare Table */
    .compare-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        background-color: #1a1e2a;
        border: 1px solid #282f42;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .compare-table th {
        background-color: #161a23;
        color: #ffffff;
        font-weight: 700;
        padding: 12px;
        text-align: center;
        border-bottom: 2px solid #1f77b4;
        font-family: 'Inter', sans-serif;
    }
    .compare-table td {
        padding: 12px;
        text-align: center;
        border-bottom: 1px solid #282f42;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    .compare-label {
        color: #8fa0c0;
        font-weight: 600;
        background-color: rgba(22, 26, 35, 0.5);
    }
    .better-stat {
        color: #2ecc71;
        font-weight: 700;
    }
    
    /* Group parent container */
    .group-box {
        background-color: #12151e;
        border: 1px solid #232837;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .group-header {
        font-size: 1.25rem;
        font-weight: bold;
        color: #ffffff;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 6px;
        margin-bottom: 15px;
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Individual match card */
    .match-card {
        background-color: #1a1e2a;
        border: 1px solid #282f42;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .match-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(31, 119, 180, 0.15);
        border-color: #1f77b4;
    }
    .match-date {
        font-size: 0.75rem;
        color: #8fa0c0;
        margin-bottom: 8px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .match-teams {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.98rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 14px;
        font-family: 'Inter', sans-serif;
        flex-wrap: wrap;
        gap: 6px;
    }
    .team-name {
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
    }
    .team-home {
        flex: 1;
        text-align: right;
        padding-right: 8px;
    }
    .team-away {
        flex: 1;
        text-align: left;
        padding-left: 8px;
    }
    .vs-badge {
        background-color: #242b3d;
        color: #ff4b4b;
        padding: 3px 8px;
        border-radius: 15px;
        font-size: 0.72rem;
        font-weight: 800;
    }
    
    /* Predictions */
    .pred-section {
        border-top: 1px solid #282f42;
        padding-top: 10px;
        margin-top: 10px;
    }
    .pred-row {
        margin-bottom: 8px;
    }
    .pred-title {
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .rf-title {
        color: #3498db;
    }
    .xgb-title {
        color: #2ecc71;
    }
    .ann-title {
        color: #f1c40f;
    }
    .pred-bars {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        padding: 6px 10px;
        border-radius: 5px;
        font-weight: 500;
        font-family: monospace;
        flex-wrap: wrap;
        gap: 6px;
    }
    .pred-bars span {
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .rf-bars {
        background-color: rgba(52, 152, 219, 0.08);
        border-left: 3px solid #3498db;
        color: #5dade2;
    }
    .xgb-bars {
        background-color: rgba(46, 204, 113, 0.08);
        border-left: 3px solid #2ecc71;
        color: #2ecc71;
    }
    .ann-bars {
        background-color: rgba(241, 196, 15, 0.08);
        border-left: 3px solid #f1c40f;
        color: #f1c40f;
    }
    
    /* Standings Table styles */
    .standings-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
        background-color: #1a1e2a;
        border: 1px solid #282f42;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .standings-table th {
        background-color: #12151e;
        color: #8fa0c0;
        font-weight: 700;
        padding: 8px 10px;
        font-size: 0.78rem;
        text-align: left;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
        border-bottom: 2px solid #1f77b4;
    }
    .standings-table td {
        padding: 8px 10px;
        color: #ffffff;
        font-size: 0.82rem;
        font-family: 'Inter', sans-serif;
        border-bottom: 1px solid #282f42;
    }
    .standings-table tr:last-child td {
        border-bottom: none;
    }
    .standings-table tr.advancing-1 {
        background-color: rgba(46, 204, 113, 0.04);
    }
    .standings-table tr.advancing-2 {
        background-color: rgba(52, 152, 219, 0.04);
    }
    .standings-table tr.advancing-3 {
        background-color: rgba(155, 89, 182, 0.04);
    }
    .standings-table tr.advancing-4 {
        background-color: rgba(231, 76, 60, 0.02);
        opacity: 0.65;
    }
    .rank-badge {
        display: inline-block;
        width: 18px;
        height: 18px;
        line-height: 18px;
        text-align: center;
        border-radius: 50%;
        font-size: 0.68rem;
        font-weight: bold;
    }
    .rank-1 { background-color: #2ecc71; color: #ffffff; }
    .rank-2 { background-color: #3498db; color: #ffffff; }
    .rank-3 { background-color: #9b59b6; color: #ffffff; }
    .rank-4 { background-color: #e74c3c; color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# Top Navigation Bar
col_logo, col_btn1, col_btn2, col_btn3 = st.columns([5, 2, 2, 2])
with col_logo:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; padding: 6px 0;">
        <img src="data:image/png;base64,{trophy_base64}" style="height: 40px; width: auto; vertical-align: middle;"/>
        <h1 style="font-size: 1.7rem; font-weight: 800; color: #ffffff; font-family: 'Inter', sans-serif; white-space: nowrap; margin: 0; padding: 0; line-height: 1.1;">
            World Cup Predictions, by Stepan
        </h1>
    </div>
    """, unsafe_allow_html=True)

with col_btn1:
    if st.button("🏠 Main", use_container_width=True, type="primary" if st.session_state.page == "Main" else "secondary"):
        st.session_state.page = "Main"
        st.rerun()

with col_btn2:
    if st.button("🔮 Make your prediction", use_container_width=True, type="primary" if st.session_state.page == "Make your prediction" else "secondary"):
        st.session_state.page = "Make your prediction"
        st.rerun()

with col_btn3:
    if st.button("🏆 Absolute predictions", use_container_width=True, type="primary" if st.session_state.page == "Absolute predictions" else "secondary"):
        st.session_state.page = "Absolute predictions"
        st.rerun()

st.markdown('<div style="border-bottom: 2px solid #1f77b4; margin-bottom: 30px; opacity: 0.8;"></div>', unsafe_allow_html=True)


# Match Dataset Raw definition
matches_raw = [
    ("11.06.2026", "Mexico", "South Africa", "Group A"),
    ("12.06.2026", "South Korea", "Czechia", "Group A"),
    ("12.06.2026", "Canada", "Bosnia", "Group B"),
    ("13.06.2026", "USA", "Paraguay", "Group D"),
    ("13.06.2026", "Qatar", "Switzerland", "Group B"),
    ("14.06.2026", "Brazil", "Morocco", "Group C"),
    ("14.06.2026", "Haiti", "Scotland", "Group C"),
    ("14.06.2026", "Australia", "Turkiye", "Group D"),
    ("14.06.2026", "Germany", "Curacao", "Group E"),
    ("14.06.2026", "Netherlands", "Japan", "Group F"),
    ("15.06.2026", "Cote Divoire", "Ecuador", "Group E"),
    ("15.06.2026", "Sweden", "Tunisia", "Group F"),
    ("15.06.2026", "Spain", "Cape Verde", "Group H"),
    ("15.06.2026", "Belgium", "Egypt", "Group G"),
    ("16.06.2026", "Saudi Arabia", "Uruguay", "Group H"),
    ("16.06.2026", "Iran", "New Zealand", "Group G"),
    ("16.06.2026", "France", "Senegal", "Group I"),
    ("17.06.2026", "Iraq", "Norway", "Group I"),
    ("17.06.2026", "Argentina", "Algeria", "Group J"),
    ("17.06.2026", "Austria", "Jordan", "Group J"),
    ("17.06.2026", "Portugal", "DR Congo", "Group K"),
    ("17.06.2026", "England", "Croatia", "Group L"),
    ("18.06.2026", "Ghana", "Panama", "Group L"),
    ("18.06.2026", "Uzbekistan", "Colombia", "Group K"),
    ("18.06.2026", "Czechia", "South Africa", "Group A"),
    ("18.06.2026", "Switzerland", "Bosnia", "Group B"),
    ("19.06.2026", "Canada", "Qatar", "Group B"),
    ("19.06.2026", "Mexico", "South Korea", "Group A"),
    ("19.06.2026", "USA", "Australia", "Group D"),
    ("20.06.2026", "Scotland", "Morocco", "Group C"),
    ("20.06.2026", "Brazil", "Haiti", "Group C"),
    ("20.06.2026", "Turkiye", "Paraguay", "Group D"),
    ("20.06.2026", "Netherlands", "Sweden", "Group F"),
    ("20.06.2026", "Germany", "Cote Divoire", "Group E"),
    ("21.06.2026", "Ecuador", "Curacao", "Group E"),
    ("21.06.2026", "Tunisia", "Japan", "Group F"),
    ("21.06.2026", "Spain", "Saudi Arabia", "Group H"),
    ("21.06.2026", "Belgium", "Iran", "Group G"),
    ("22.06.2026", "Uruguay", "Cape Verde", "Group H"),
    ("22.06.2026", "New Zealand", "Egypt", "Group G"),
    ("22.06.2026", "Argentina", "Austria", "Group J"),
    ("23.06.2026", "France", "Iraq", "Group I"),
    ("23.06.2026", "Norway", "Senegal", "Group I"),
    ("23.06.2026", "Jordan", "Algeria", "Group J"),
    ("23.06.2026", "Portugal", "Uzbekistan", "Group K"),
    ("23.06.2026", "England", "Ghana", "Group L"),
    ("24.06.2026", "Panama", "Croatia", "Group L"),
    ("24.06.2026", "Colombia", "DR Congo", "Group K"),
    ("24.06.2026", "Switzerland", "Canada", "Group B"),
    ("24.06.2026", "Bosnia", "Qatar", "Group B"),
    ("25.06.2026", "Scotland", "Brazil", "Group C"),
    ("25.06.2026", "Morocco", "Haiti", "Group C"),
    ("25.06.2026", "Czechia", "Mexico", "Group A"),
    ("25.06.2026", "South Africa", "South Korea", "Group A"),
    ("25.06.2026", "Ecuador", "Germany", "Group E"),
    ("25.06.2026", "Curacao", "Cote Divoire", "Group E"),
    ("26.06.2026", "Tunisia", "Netherlands", "Group F"),
    ("26.06.2026", "Japan", "Sweden", "Group F"),
    ("26.06.2026", "Turkiye", "USA", "Group D"),
    ("26.06.2026", "Paraguay", "Australia", "Group D"),
    ("26.06.2026", "Norway", "France", "Group I"),
    ("26.06.2026", "Senegal", "Iraq", "Group I"),
    ("27.06.2026", "Uruguay", "Spain", "Group H"),
    ("27.06.2026", "Cape Verde", "Saudi Arabia", "Group H"),
    ("27.06.2026", "Egypt", "Iran", "Group G"),
    ("27.06.2026", "New Zealand", "Belgium", "Group G"),
    ("28.06.2026", "Panama", "England", "Group L"),
    ("28.06.2026", "Croatia", "Ghana", "Group L"),
    ("28.06.2026", "Colombia", "Portugal", "Group K"),
    ("28.06.2026", "DR Congo", "Uzbekistan", "Group K"),
    ("28.06.2026", "Jordan", "Argentina", "Group J"),
    ("28.06.2026", "Algeria", "Austria", "Group J")
]



# Parse and Sort Matches by Date
matches_by_date = {}
for date_str, t1, t2, grp in matches_raw:
    dt = datetime.strptime(date_str, "%d.%m.%Y")
    matches_by_date.setdefault(dt, []).append((date_str, t1, t2, grp))

# Sort dates: chronological order (most recent to later)
sorted_dates = sorted(matches_by_date.keys())


# Page Routing
if st.session_state.page == "Main":
    st.markdown("### 🏆 World Cup 2026 - Group Stage Predictions")
    st.write("Below are the match schedules and prediction probabilities generated from the machine learning models. Tap or hover over any match card to highlight.")

    # Display date sections
    for dt in sorted_dates:
        date_str_formatted = dt.strftime("%A, %d %B %Y")
        matches = matches_by_date[dt]
        
        st.markdown(f"#### 📅 {date_str_formatted}")
        
        # Group matches on this date by their Group
        groups_on_date = {}
        for date_str, t1, t2, grp in matches:
            groups_on_date.setdefault(grp, []).append((t1, t2))
            
        # Render Groups in columns
        cols = st.columns(len(groups_on_date))
        for col, (grp_name, pairings) in zip(cols, groups_on_date.items()):
            with col:
                # Begin parent rectangle Group box
                group_html = f'<div class="group-box"><div class="group-header">⚽ {grp_name}</div>'
                
                # Loop through match pairings inside the Group Box
                for t1, t2 in pairings:
                    # Get model predictions
                    probs = predict_match_probs(t1, t2)
                    rf_t1, rf_draw, rf_t2 = probs["rf"]
                    xgb_t1, xgb_draw, xgb_t2 = probs["xgb"]
                    ann_t1, ann_draw, ann_t2 = probs["ann"]
                    
                    # Fetch flags base64
                    t1_flag = flag_file_map.get(t1, "")
                    t2_flag = flag_file_map.get(t2, "")
                    
                    t1_path = os.path.join(project_root, "Images", "Flags", t1_flag)
                    t2_path = os.path.join(project_root, "Images", "Flags", t2_flag)
                    
                    t1_b64 = get_base64_image(t1_path)
                    t2_b64 = get_base64_image(t2_path)
                    
                    ext1 = t1_flag.split('.')[-1]
                    ext2 = t2_flag.split('.')[-1]
                    ext1 = "jpeg" if ext1 == "jpg" else ext1
                    ext2 = "jpeg" if ext2 == "jpg" else ext2
                    
                    t1_img = f'<img src="data:image/{ext1};base64,{t1_b64}" style="height: 14px; width: auto; vertical-align: middle; margin-left: 6px; border-radius: 3px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);"/>' if t1_b64 else ''
                    t2_img = f'<img src="data:image/{ext2};base64,{t2_b64}" style="height: 14px; width: auto; vertical-align: middle; margin-right: 6px; border-radius: 3px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);"/>' if t2_b64 else ''
                    
                    # Build match card HTML
                    group_html += f"""<div class="match-card">
<div class="match-teams">
<div class="team-name team-home">{t1} {t1_img}</div>
<div class="vs-badge">VS</div>
<div class="team-name team-away">{t2_img} {t2}</div>
</div>
<div class="pred-section">
<div class="pred-row">
<div class="pred-title rf-title">🔵 Prediction 1 (Random Forest)</div>
<div class="pred-bars rf-bars">
<span>{t1}: {rf_t1:.1f}%</span>
<span>Draw: {rf_draw:.1f}%</span>
<span>{t2}: {rf_t2:.1f}%</span>
</div>
</div>
<div class="pred-row">
<div class="pred-title xgb-title">🟢 Prediction 2 (XGBoost)</div>
<div class="pred-bars xgb-bars">
<span>{t1}: {xgb_t1:.1f}%</span>
<span>Draw: {xgb_draw:.1f}%</span>
<span>{t2}: {xgb_t2:.1f}%</span>
</div>
</div>
<div class="pred-row">
<div class="pred-title ann-title">🟡 Prediction 3 (Neural Network)</div>
<div class="pred-bars ann-bars">
<span>{t1}: {ann_t1:.1f}%</span>
<span>Draw: {ann_draw:.1f}%</span>
<span>{t2}: {ann_t2:.1f}%</span>
</div>
</div>
</div>
</div>"""
                
                # Close parent rectangle
                group_html += "</div>"
                st.markdown(group_html, unsafe_allow_html=True)

elif st.session_state.page == "Make your prediction":
    st.markdown("### 🔮 Custom Matchup Predictor")
    st.write("Simulate any matchup between the 48 qualified World Cup 2026 teams. Our machine learning models will estimate the probability of each outcome based on historical performance, ELO ratings, and recent statistics.")
    
    # Deduplicate team list
    all_teams_list = sorted(list(set(
        "Cote Divoire" if t == "Cote divoire" else t for t in flag_file_map.keys()
    )))
    
    # Select default teams
    default_home = "Argentina"
    default_away = "France"
    if default_home not in all_teams_list:
        default_home = all_teams_list[0]
    if default_away not in all_teams_list:
        default_away = all_teams_list[min(len(all_teams_list)-1, 1)]
        
    col_t1, col_vs, col_t2 = st.columns([5, 1, 5])
    with col_t1:
        team1 = st.selectbox("Select Team 1 (Home/Neutral)", all_teams_list, index=all_teams_list.index(default_home))
    with col_vs:
        st.markdown("<div style='text-align: center; font-size: 2.2rem; font-weight: 800; color: #ff4b4b; margin-top: 25px;'>VS</div>", unsafe_allow_html=True)
    with col_t2:
        team2 = st.selectbox("Select Team 2 (Away/Neutral)", all_teams_list, index=all_teams_list.index(default_away))
        
    if team1 == team2:
        st.warning("⚠️ Please select two different teams to run the match simulation.")
    else:
        # Run simulation
        probs = predict_match_probs(team1, team2)
        rf_t1, rf_draw, rf_t2 = probs["rf"]
        xgb_t1, xgb_draw, xgb_t2 = probs["xgb"]
        ann_t1, ann_draw, ann_t2 = probs["ann"]
        
        # Fetch flags base64
        t1_flag = flag_file_map.get(team1, "")
        t2_flag = flag_file_map.get(team2, "")
        
        t1_path = os.path.join(project_root, "Images", "Flags", t1_flag)
        t2_path = os.path.join(project_root, "Images", "Flags", t2_flag)
        
        t1_b64 = get_base64_image(t1_path)
        t2_b64 = get_base64_image(t2_path)
        
        ext1 = t1_flag.split('.')[-1] if t1_flag else "png"
        ext2 = t2_flag.split('.')[-1] if t2_flag else "png"
        ext1 = "jpeg" if ext1 == "jpg" else ext1
        ext2 = "jpeg" if ext2 == "jpg" else ext2
        
        t1_img = f'<img src="data:image/{ext1};base64,{t1_b64}" style="height: 24px; width: auto; vertical-align: middle; margin-left: 10px; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.4);"/>' if t1_b64 else ''
        t2_img = f'<img src="data:image/{ext2};base64,{t2_b64}" style="height: 24px; width: auto; vertical-align: middle; margin-right: 10px; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.4);"/>' if t2_b64 else ''
        
        # Render Match Card
        st.markdown(f"""<div style="max-width: 800px; margin: 0 auto; background-color: #12151e; border: 1px solid #232837; border-radius: 12px; padding: 25px; box-shadow: 0 6px 16px rgba(0,0,0,0.4);">
<div style="font-size: 0.95rem; color: #8fa0c0; text-align: center; margin-bottom: 15px; font-weight: 600;">⚽ SIMULATED MATCHUP</div>
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 1.6rem; font-weight: 800; color: #ffffff; margin-bottom: 25px; font-family: 'Inter', sans-serif;">
<div style="flex: 1; text-align: right; padding-right: 15px; display: flex; align-items: center; justify-content: flex-end; gap: 10px;">
<span>{team1}</span>
{t1_img}
</div>
<div class="vs-badge" style="font-size: 0.9rem; padding: 5px 12px; border-radius: 20px;">VS</div>
<div style="flex: 1; text-align: left; padding-left: 15px; display: flex; align-items: center; justify-content: flex-start; gap: 10px;">
{t2_img}
<span>{team2}</span>
</div>
</div>
<div class="pred-section" style="border-top: 1px solid #232837; padding-top: 20px;">
<div class="pred-row" style="margin-bottom: 18px;">
<div class="pred-title rf-title" style="font-size: 0.9rem; margin-bottom: 8px;">🔵 Prediction 1: Random Forest Classifier</div>
<div class="pred-bars rf-bars" style="font-size: 0.9rem; padding: 10px 15px; border-radius: 6px;">
<span>{team1}: <strong>{rf_t1:.1f}%</strong></span>
<span>Draw: <strong>{rf_draw:.1f}%</strong></span>
<span>{team2}: <strong>{rf_t2:.1f}%</strong></span>
</div>
</div>
<div class="pred-row" style="margin-bottom: 18px;">
<div class="pred-title xgb-title" style="font-size: 0.9rem; margin-bottom: 8px;">🟢 Prediction 2: XGBoost Classifier (Calibrated)</div>
<div class="pred-bars xgb-bars" style="font-size: 0.9rem; padding: 10px 15px; border-radius: 6px;">
<span>{team1}: <strong>{xgb_t1:.1f}%</strong></span>
<span>Draw: <strong>{xgb_draw:.1f}%</strong></span>
<span>{team2}: <strong>{xgb_t2:.1f}%</strong></span>
</div>
</div>
<div class="pred-row">
<div class="pred-title ann-title" style="font-size: 0.9rem; margin-bottom: 8px;">🟡 Prediction 3: Artificial Neural Network</div>
<div class="pred-bars ann-bars" style="font-size: 0.9rem; padding: 10px 15px; border-radius: 6px;">
<span>{team1}: <strong>{ann_t1:.1f}%</strong></span>
<span>Draw: <strong>{ann_draw:.1f}%</strong></span>
<span>{team2}: <strong>{ann_t2:.1f}%</strong></span>
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

elif st.session_state.page == "Absolute predictions":
    st.markdown("### 🏆 Absolute World Cup 2026 Predictions")
    st.write("This page simulates the entire World Cup 2026 tournament, from the group stages to the grand final. Predictions are generated dynamically using our machine learning models.")
    
    # Run the tournament simulation
    sim = simulate_world_cup(matches_raw)
    
    # Setup sub-tabs for Group standings vs Knockout bracket
    tab_standings, tab_knockout = st.tabs(["📊 Group Standings", "🌳 Knockout Bracket"])
    
    with tab_standings:
        st.write("#### ⚽ Predicted Group Stage Tables")
        st.write("The top 2 teams from each group and the 8 best third-placed teams advance to the Round of 32. Level teams are ranked by Point score, with Team ELO used as a tiebreaker.")
        
        # Render groups in columns
        group_names = sorted(list(sim["ranked_groups"].keys()))
        # Render groups A-L in a 3-column layout (4 rows)
        for i in range(0, len(group_names), 3):
            cols = st.columns(3)
            for j, grp in enumerate(group_names[i:i+3]):
                with cols[j]:
                    st.markdown(f"<div style='margin-top: 15px; font-size: 1.15rem; font-weight: bold; color: #ffffff;'>⚽ {grp}</div>", unsafe_allow_html=True)
                    table_html = render_group_standings_table(grp, sim["ranked_groups"][grp], {t["group"] for t in sim["advancing_third_places"]})
                    st.markdown(table_html, unsafe_allow_html=True)
                    
        st.markdown("<hr style='border-color: #282f42;'/>", unsafe_allow_html=True)
        st.write("#### 🥉 Third-Placed Teams Standings")
        st.write("All 12 third-placed teams ranked. The top 8 qualify for the Round of 32 and are assigned to their opponents via a dynamic Annex C matching algorithm.")
        
        third_places_html = render_third_places_table(sim["third_places"], {t["group"] for t in sim["advancing_third_places"]})
        st.markdown(third_places_html, unsafe_allow_html=True)
        
    with tab_knockout:
        # Sub-tabs for each knockout round to keep it extremely clean
        tab_r32, tab_r16, tab_qf, tab_sf, tab_finals = st.tabs([
            "Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "🏆 Finals & Champion"
        ])
        
        with tab_r32:
            st.write("#### Round of 32 Fixtures & Predictions")
            r32_results = sim["r32_results"]
            r32_ids = sorted(list(r32_results.keys()))
            
            # Show in a 4-column grid (4 rows of 4 cards)
            for r_start in range(0, len(r32_ids), 4):
                cols = st.columns(4)
                for c_idx, match_id in enumerate(r32_ids[r_start:r_start+4]):
                    with cols[c_idx]:
                        match_html = render_knockout_match_card(r32_results[match_id])
                        st.markdown(match_html, unsafe_allow_html=True)
                        
        with tab_r16:
            st.write("#### Round of 16 Fixtures & Predictions")
            r16_results = sim["r16_results"]
            r16_ids = sorted(list(r16_results.keys()))
            
            # Show in a 4-column grid (2 rows of 4 cards)
            for r_start in range(0, len(r16_ids), 4):
                cols = st.columns(4)
                for c_idx, match_id in enumerate(r16_ids[r_start:r_start+4]):
                    with cols[c_idx]:
                        match_html = render_knockout_match_card(r16_results[match_id])
                        st.markdown(match_html, unsafe_allow_html=True)
                        
        with tab_qf:
            st.write("#### Quarter-Final Fixtures & Predictions")
            qf_results = sim["qf_results"]
            qf_ids = sorted(list(qf_results.keys()))
            
            # Show in a 4-column grid (1 row of 4 cards)
            cols = st.columns(4)
            for c_idx, match_id in enumerate(qf_ids):
                with cols[c_idx]:
                    match_html = render_knockout_match_card(qf_results[match_id])
                    st.markdown(match_html, unsafe_allow_html=True)
                    
        with tab_sf:
            st.write("#### Semi-Final Fixtures & Predictions")
            sf_results = sim["sf_results"]
            sf_ids = sorted(list(sf_results.keys()))
            
            # Show in 2 columns
            cols = st.columns(2)
            for c_idx, match_id in enumerate(sf_ids):
                with cols[c_idx]:
                    match_html = render_knockout_match_card(sf_results[match_id])
                    st.markdown(match_html, unsafe_allow_html=True)
                    
        with tab_finals:
            champion = sim["final_result"]["winner"]
            
            # Render Champion Showcase
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 30px; padding: 25px; background-color: rgba(46, 204, 113, 0.08); border: 2px solid #2ecc71; border-radius: 12px; max-width: 600px; margin: 20px auto;">
                <img src="data:image/png;base64,{trophy_base64}" style="height: 100px; width: auto; margin-bottom: 15px;"/>
                <h2 style="color: #ffffff; margin: 0; font-family: 'Inter', sans-serif; letter-spacing: 1px;">WORLD CUP 2026 CHAMPION</h2>
                <h1 style="color: #2ecc71; margin: 15px 0; font-size: 2.8rem; font-weight: 800; font-family: 'Inter', sans-serif; text-transform: uppercase;">
                    {champion}
                </h1>
                <span style="color: #8fa0c0; font-size: 0.9rem; font-weight: 600;">
                    Simulated using Random Forest, XGBoost, and Artificial Neural Network predictions
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Render Final and Bronze Final side by side
            col_final, col_bronze = st.columns(2)
            with col_final:
                st.write("<h4 style='text-align: center; color: #2ecc71; margin-bottom: 15px;'>🥇 Grand Final Prediction</h4>", unsafe_allow_html=True)
                match_html = render_knockout_match_card(sim["final_result"])
                st.markdown(match_html, unsafe_allow_html=True)
            with col_bronze:
                st.write("<h4 style='text-align: center; color: #5dade2; margin-bottom: 15px;'>🥉 Bronze Final Prediction</h4>", unsafe_allow_html=True)
                match_html = render_knockout_match_card(sim["bronze_result"])
                st.markdown(match_html, unsafe_allow_html=True)


