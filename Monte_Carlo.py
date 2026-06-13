import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ETL.Transform.feature_engineering import preprocessing

models_dir = os.path.join(project_root, "src", "Models", "Dumped_models")
CT = joblib.load(os.path.join(models_dir, "ColumnTransformer.pkl"))
imputer = joblib.load(os.path.join(models_dir, "SimpleImputer.pkl"))
sc = joblib.load(os.path.join(models_dir, "StandardScaler.pkl"))

rf_model = joblib.load(os.path.join(models_dir, "RandomForest_Classifier.pkl"))
xgb_model = joblib.load(os.path.join(models_dir, "XGBoost_Classifier.pkl"))
ann_model = joblib.load(os.path.join(models_dir, "Neural_Network.pkl"))

raw_data_path = os.path.join(project_root, "Data", "raw", "nations_data.csv")
messy_dataset = pd.read_csv(raw_data_path)
dataset, stats, team_elos = preprocessing(messy_dataset)
median_stats = stats.median(numeric_only=True)

normalization_map = {
    "Bosnia": "Bosnia & Herzegovina",
    "Turkiye": "Türkiye",
    "Curacao": "Curaçao",
    "Cote Divoire": "Côte d'Ivoire",
    "Cote divoire": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde"
}

def get_normalized_name(team):
    return normalization_map.get(team, team)

def get_team_features_row(team_name):
    matching = stats[stats['Team'] == team_name]
    if len(matching) > 0:
        return matching.iloc[-1]
    return median_stats

def get_match_features_df(home_team, away_team):
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

def predict_match_probs_mc(team1, team2):
    """Calculates averaged Home/Away probabilities for RF, XGBoost, and ANN."""
    h_norm = get_normalized_name(team1)
    a_norm = get_normalized_name(team2)
    
    fwd_df = get_match_features_df(h_norm, a_norm)
    fwd_trans = CT.transform(fwd_df)
    fwd_imp = imputer.transform(fwd_trans)
    fwd_scaled = sc.transform(fwd_imp)
    
    rf_probs_fwd = rf_model.predict_proba(fwd_imp)[0]
    xgb_probs_fwd = xgb_model.predict_proba(fwd_imp)[0]
    ann_probs_fwd = ann_model.predict_proba(fwd_scaled)[0]
    
    bwd_df = get_match_features_df(a_norm, h_norm)
    bwd_trans = CT.transform(bwd_df)
    bwd_imp = imputer.transform(bwd_trans)
    bwd_scaled = sc.transform(bwd_imp)
    
    rf_probs_bwd = rf_model.predict_proba(bwd_imp)[0]
    xgb_probs_bwd = xgb_model.predict_proba(bwd_imp)[0]
    ann_probs_bwd = ann_model.predict_proba(bwd_scaled)[0]
    
    rf_t1_win = (rf_probs_fwd[2] + rf_probs_bwd[0]) / 2.0
    rf_draw   = (rf_probs_fwd[1] + rf_probs_bwd[1]) / 2.0
    rf_t2_win = (rf_probs_fwd[0] + rf_probs_bwd[2]) / 2.0
    
    xgb_t1_win = (xgb_probs_fwd[2] + xgb_probs_bwd[0]) / 2.0
    xgb_draw   = (xgb_probs_fwd[1] + xgb_probs_bwd[1]) / 2.0
    xgb_t2_win = (xgb_probs_fwd[0] + xgb_probs_bwd[2]) / 2.0
    ann_t1_win = (ann_probs_fwd[2] + ann_probs_bwd[0]) / 2.0
    ann_draw   = (ann_probs_fwd[1] + ann_probs_bwd[1]) / 2.0
    ann_t2_win = (ann_probs_fwd[0] + ann_probs_bwd[2]) / 2.0
    
    return {
        "rf": (rf_t1_win, rf_draw, rf_t2_win),
        "xgb": (xgb_t1_win, xgb_draw, xgb_t2_win),
        "ann": (ann_t1_win, ann_draw, ann_t2_win)
    }

def resolve_draws_vectorized(team1, team2, n_draws):
    """
    Resolves draws at knockout stages using a smaller Monte Carlo simulation of 10 times.
    The team with higher ELO has a 60% chance of winning, the other has 40%.
    """
    h_norm = get_normalized_name(team1)
    a_norm = get_normalized_name(team2)
    
    elo1 = team_elos.get(h_norm, 1500)
    elo2 = team_elos.get(a_norm, 1500)
    
    if elo1 > elo2:
        wins = np.random.binomial(10, 0.6, size=n_draws)
        outcomes = np.empty(n_draws, dtype=int)
        outcomes[wins >= 6] = 2 
        outcomes[wins <= 4] = 0 
        
        ties = (wins == 5)
        num_ties = np.sum(ties)
        if num_ties > 0:
            outcomes[ties] = np.random.choice([2, 0], size=num_ties)
        return outcomes
    elif elo2 > elo1:
        wins = np.random.binomial(10, 0.6, size=n_draws)
        outcomes = np.empty(n_draws, dtype=int)
        outcomes[wins >= 6] = 0 
        outcomes[wins <= 4] = 2 
        
        ties = (wins == 5)
        num_ties = np.sum(ties)
        if num_ties > 0:
            outcomes[ties] = np.random.choice([0, 2], size=num_ties)
        return outcomes
    else:
        # Equal Elo: 50/50 chance
        return np.random.choice([2, 0], size=n_draws)

def run_monte_carlo_for_match(team1, team2, trials=10000):
    """
    Runs a Monte Carlo simulation 10,000 times for each model.
    Returns: {
        "rf": (rf_t1, rf_draw, rf_t2),
        "xgb": (xgb_t1, xgb_draw, xgb_t2),
        "ann": (ann_t1, ann_draw, ann_t2)
    }
    """
    probs = predict_match_probs_mc(team1, team2)
    
    rf_p = np.array(probs["rf"])
    xgb_p = np.array(probs["xgb"])
    ann_p = np.array(probs["ann"])
    
    rf_p /= rf_p.sum()
    xgb_p /= xgb_p.sum()
    ann_p /= ann_p.sum()
    
    # Map index 0 (T1 win) to outcome 2, index 1 (Draw) to outcome 1, index 2 (T2 win) to outcome 0
    p_rf = [rf_p[2], rf_p[1], rf_p[0]]
    p_xgb = [xgb_p[2], xgb_p[1], xgb_p[0]]
    p_ann = [ann_p[2], ann_p[1], ann_p[0]]
    
    rf_choices = np.random.choice([0, 1, 2], size=trials, p=p_rf)
    xgb_choices = np.random.choice([0, 1, 2], size=trials, p=p_xgb)
    ann_choices = np.random.choice([0, 1, 2], size=trials, p=p_ann)
    
    rf_t1 = (np.sum(rf_choices == 2) / trials) * 100.0
    rf_draw = (np.sum(rf_choices == 1) / trials) * 100.0
    rf_t2 = (np.sum(rf_choices == 0) / trials) * 100.0
    
    xgb_t1 = (np.sum(xgb_choices == 2) / trials) * 100.0
    xgb_draw = (np.sum(xgb_choices == 1) / trials) * 100.0
    xgb_t2 = (np.sum(xgb_choices == 0) / trials) * 100.0
    
    ann_t1 = (np.sum(ann_choices == 2) / trials) * 100.0
    ann_draw = (np.sum(ann_choices == 1) / trials) * 100.0
    ann_t2 = (np.sum(ann_choices == 0) / trials) * 100.0
    
    return {
        "rf": (rf_t1, rf_draw, rf_t2),
        "xgb": (xgb_t1, xgb_draw, xgb_t2),
        "ann": (ann_t1, ann_draw, ann_t2),
        "rf_choices": rf_choices,
        "xgb_choices": xgb_choices,
        "ann_choices": ann_choices
    }

def get_model_prediction(team1, team2, t1_prob, draw_prob, t2_prob, is_knockout=None):
    """
    Determines model prediction from simulated probabilities.
    A model predicts a win, or a draw if the respective probability is the maximum of the 3 probabilities.
    Returns outcome: 2 (T1 win), 1 (Draw), 0 (T2 win)
    """
    max_prob = max(t1_prob, draw_prob, t2_prob)
    if draw_prob == max_prob:
        return 1
    elif t1_prob == t2_prob and t1_prob == max_prob:
        return 1
    elif t1_prob == max_prob:
        return 2
    else:
        return 0

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

def save_raw_trials_to_csv(filepath, stage, match_id, date, team1, team2, mc_res):
    mapping = {2: team1, 1: "Draw", 0: team2}
    trials_per_model = len(mc_res["rf_choices"])
    
    models = np.concatenate([
        np.full(trials_per_model, "Random Forest", dtype=object),
        np.full(trials_per_model, "XGBoost", dtype=object),
        np.full(trials_per_model, "Neural Network", dtype=object)
    ])
    
    choices = np.concatenate([
        mc_res["rf_choices"],
        mc_res["xgb_choices"],
        mc_res["ann_choices"]
    ])
    
    total_rows = trials_per_model * 3
    t1_arr = np.full(total_rows, team1, dtype=object)
    t2_arr = np.full(total_rows, team2, dtype=object)
    stage_arr = np.full(total_rows, stage, dtype=object)
    match_id_arr = np.full(total_rows, str(match_id) if match_id is not None else "", dtype=object)
    date_arr = np.full(total_rows, date, dtype=object)
    
    trial_indices = np.tile(np.arange(1, trials_per_model + 1), 3)
    outcomes = [mapping[c] for c in choices]
    
    df = pd.DataFrame({
        "Stage": stage_arr,
        "Match_ID": match_id_arr,
        "Date": date_arr,
        "Team1": t1_arr,
        "Team2": t2_arr,
        "Model": models,
        "Trial_Index": trial_indices,
        "Simulated_Outcome": outcomes
    })
    
    df.to_csv(filepath, mode='a', header=False, index=False)

def simulate_world_cup_monte_carlo(matches_list, raw_csv_path=None):
    if raw_csv_path:
        os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
        pd.DataFrame(columns=["Stage", "Match_ID", "Date", "Team1", "Team2", "Model", "Trial_Index", "Simulated_Outcome"]).to_csv(raw_csv_path, index=False)

    # Group Stage initialization
    groups = {}
    for _, t1, t2, grp in matches_list:
        if grp not in groups:
            groups[grp] = {}
        if t1 not in groups[grp]:
            groups[grp][t1] = {"team": t1, "points": 0, "wins": 0, "draws": 0, "losses": 0}
        if t2 not in groups[grp]:
            groups[grp][t2] = {"team": t2, "points": 0, "wins": 0, "draws": 0, "losses": 0}
            
    match_records = []
    
    # Simulate Group Stage
    for date_str, t1, t2, grp in matches_list:
        mc_res = run_monte_carlo_for_match(t1, t2)
        if raw_csv_path:
            save_raw_trials_to_csv(raw_csv_path, grp, "", date_str, t1, t2, mc_res)
        rf_t1, rf_draw, rf_t2 = mc_res["rf"]
        xgb_t1, xgb_draw, xgb_t2 = mc_res["xgb"]
        ann_t1, ann_draw, ann_t2 = mc_res["ann"]
        
        rf_pred = get_model_prediction(t1, t2, rf_t1, rf_draw, rf_t2, is_knockout=False)
        xgb_pred = get_model_prediction(t1, t2, xgb_t1, xgb_draw, xgb_t2, is_knockout=False)
        ann_pred = get_model_prediction(t1, t2, ann_t1, ann_draw, ann_t2, is_knockout=False)
        
        votes = [rf_pred, xgb_pred, ann_pred]
        counts = {c: votes.count(c) for c in [2, 1, 0]}
        
        majority_result = None
        for c, count in counts.items():
            if count >= 2:
                majority_result = c
                break
                
        if majority_result is None:
            p1 = (rf_t1 + xgb_t1 + ann_t1) / 3.0
            p_draw = (rf_draw + xgb_draw + ann_draw) / 3.0
            p2 = (rf_t2 + xgb_t2 + ann_t2) / 3.0
            if p1 > p_draw and p1 > p2:
                majority_result = 2
            elif p2 > p_draw and p2 > p1:
                majority_result = 0
            else:
                majority_result = 1
                
        if majority_result == 2:
            outcome = 'H'
            winner = t1
            loser = t2
        elif majority_result == 0:
            outcome = 'A'
            winner = t2
            loser = t1
        else:
            outcome = 'D'
            winner = 'Draw'
            loser = 'Draw'
            
        if outcome == 'H':
            groups[grp][t1]["points"] += 3
            groups[grp][t1]["wins"] += 1
            groups[grp][t2]["losses"] += 1
        elif outcome == 'A':
            groups[grp][t2]["points"] += 3
            groups[grp][t2]["wins"] += 1
            groups[grp][t1]["losses"] += 1
        else:
            groups[grp][t1]["points"] += 1
            groups[grp][t1]["draws"] += 1
            groups[grp][t2]["points"] += 1
            groups[grp][t2]["draws"] += 1
            
        match_records.append({
            "Stage": grp,
            "Match_ID": "",
            "Date": date_str,
            "Team1": t1,
            "Team2": t2,
            "Prob_T1": (rf_t1 + xgb_t1 + ann_t1) / 3.0,
            "Prob_Draw": (rf_draw + xgb_draw + ann_draw) / 3.0,
            "Prob_T2": (rf_t2 + xgb_t2 + ann_t2) / 3.0,
            "RF_Prob_T1": rf_t1,
            "RF_Prob_Draw": rf_draw,
            "RF_Prob_T2": rf_t2,
            "XGB_Prob_T1": xgb_t1,
            "XGB_Prob_Draw": xgb_draw,
            "XGB_Prob_T2": xgb_t2,
            "ANN_Prob_T1": ann_t1,
            "ANN_Prob_Draw": ann_draw,
            "ANN_Prob_T2": ann_t2,
            "Winner": winner,
            "Loser": loser
        })
        
    # Rank groups
    ranked_groups = {}
    for grp, teams_dict in groups.items():
        teams_list = list(teams_dict.values())
        teams_list.sort(key=lambda x: (x["points"], team_elos.get(get_normalized_name(x["team"]), 1500)), reverse=True)
        ranked_groups[grp] = teams_list
        
    third_places = []
    for grp, teams_list in ranked_groups.items():
        grp_letter = grp.replace("Group ", "")
        third_team = teams_list[2].copy()
        third_team["group"] = grp_letter
        third_places.append(third_team)
        
    third_places.sort(key=lambda x: (x["points"], team_elos.get(get_normalized_name(x["team"]), 1500)), reverse=True)
    
    advancing_third_places = third_places[:8]
    advancing_third_groups = {item["group"] for item in advancing_third_places}
    third_place_mapping = find_third_place_matching(advancing_third_groups)
    
    # R32 Simulation
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
        t1, t2 = fixture["t1"], fixture["t2"]
        mc_res = run_monte_carlo_for_match(t1, t2)
        if raw_csv_path:
            save_raw_trials_to_csv(raw_csv_path, "Round of 32", fixture["id"], fixture["date"], t1, t2, mc_res)
        rf_t1, rf_draw, rf_t2 = mc_res["rf"]
        xgb_t1, xgb_draw, xgb_t2 = mc_res["xgb"]
        ann_t1, ann_draw, ann_t2 = mc_res["ann"]
        
        rf_pred = get_model_prediction(t1, t2, rf_t1, rf_draw, rf_t2, is_knockout=True)
        xgb_pred = get_model_prediction(t1, t2, xgb_t1, xgb_draw, xgb_t2, is_knockout=True)
        ann_pred = get_model_prediction(t1, t2, ann_t1, ann_draw, ann_t2, is_knockout=True)
        
        votes = [rf_pred, xgb_pred, ann_pred]
        if votes.count(2) >= 2:
            winner = t1
        elif votes.count(0) >= 2:
            winner = t2
        else:
            resolved = resolve_draws_vectorized(t1, t2, 1)[0]
            winner = t1 if resolved == 2 else t2
        loser = t2 if winner == t1 else t1
        
        r32_results[fixture["id"]] = {
            "id": fixture["id"],
            "date": fixture["date"],
            "stadium": fixture["stadium"],
            "t1": t1,
            "t2": t2,
            "winner": winner,
            "loser": loser,
            "probs": mc_res
        }
        
        match_records.append({
            "Stage": "Round of 32",
            "Match_ID": fixture["id"],
            "Date": fixture["date"],
            "Team1": t1,
            "Team2": t2,
            "Prob_T1": (rf_t1 + xgb_t1 + ann_t1) / 3.0,
            "Prob_Draw": (rf_draw + xgb_draw + ann_draw) / 3.0,
            "Prob_T2": (rf_t2 + xgb_t2 + ann_t2) / 3.0,
            "RF_Prob_T1": rf_t1,
            "RF_Prob_Draw": rf_draw,
            "RF_Prob_T2": rf_t2,
            "XGB_Prob_T1": xgb_t1,
            "XGB_Prob_Draw": xgb_draw,
            "XGB_Prob_T2": xgb_t2,
            "ANN_Prob_T1": ann_t1,
            "ANN_Prob_Draw": ann_draw,
            "ANN_Prob_T2": ann_t2,
            "Winner": winner,
            "Loser": loser
        })
        
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
        mc_res = run_monte_carlo_for_match(t1, t2)
        if raw_csv_path:
            save_raw_trials_to_csv(raw_csv_path, "Round of 16", fixture["id"], fixture["date"], t1, t2, mc_res)
        rf_t1, rf_draw, rf_t2 = mc_res["rf"]
        xgb_t1, xgb_draw, xgb_t2 = mc_res["xgb"]
        ann_t1, ann_draw, ann_t2 = mc_res["ann"]
        
        rf_pred = get_model_prediction(t1, t2, rf_t1, rf_draw, rf_t2, is_knockout=True)
        xgb_pred = get_model_prediction(t1, t2, xgb_t1, xgb_draw, xgb_t2, is_knockout=True)
        ann_pred = get_model_prediction(t1, t2, ann_t1, ann_draw, ann_t2, is_knockout=True)
        
        votes = [rf_pred, xgb_pred, ann_pred]
        if votes.count(2) >= 2:
            winner = t1
        elif votes.count(0) >= 2:
            winner = t2
        else:
            resolved = resolve_draws_vectorized(t1, t2, 1)[0]
            winner = t1 if resolved == 2 else t2
        loser = t2 if winner == t1 else t1
        
        r16_results[fixture["id"]] = {
            "id": fixture["id"],
            "date": fixture["date"],
            "stadium": fixture["stadium"],
            "t1": t1,
            "t2": t2,
            "winner": winner,
            "loser": loser,
            "probs": mc_res
        }
        
        match_records.append({
            "Stage": "Round of 16",
            "Match_ID": fixture["id"],
            "Date": fixture["date"],
            "Team1": t1,
            "Team2": t2,
            "Prob_T1": (rf_t1 + xgb_t1 + ann_t1) / 3.0,
            "Prob_Draw": (rf_draw + xgb_draw + ann_draw) / 3.0,
            "Prob_T2": (rf_t2 + xgb_t2 + ann_t2) / 3.0,
            "RF_Prob_T1": rf_t1,
            "RF_Prob_Draw": rf_draw,
            "RF_Prob_T2": rf_t2,
            "XGB_Prob_T1": xgb_t1,
            "XGB_Prob_Draw": xgb_draw,
            "XGB_Prob_T2": xgb_t2,
            "ANN_Prob_T1": ann_t1,
            "ANN_Prob_Draw": ann_draw,
            "ANN_Prob_T2": ann_t2,
            "Winner": winner,
            "Loser": loser
        })
        
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
        mc_res = run_monte_carlo_for_match(t1, t2)
        if raw_csv_path:
            save_raw_trials_to_csv(raw_csv_path, "Quarter-Finals", fixture["id"], fixture["date"], t1, t2, mc_res)
        rf_t1, rf_draw, rf_t2 = mc_res["rf"]
        xgb_t1, xgb_draw, xgb_t2 = mc_res["xgb"]
        ann_t1, ann_draw, ann_t2 = mc_res["ann"]
        
        rf_pred = get_model_prediction(t1, t2, rf_t1, rf_draw, rf_t2, is_knockout=True)
        xgb_pred = get_model_prediction(t1, t2, xgb_t1, xgb_draw, xgb_t2, is_knockout=True)
        ann_pred = get_model_prediction(t1, t2, ann_t1, ann_draw, ann_t2, is_knockout=True)
        
        votes = [rf_pred, xgb_pred, ann_pred]
        if votes.count(2) >= 2:
            winner = t1
        elif votes.count(0) >= 2:
            winner = t2
        else:
            resolved = resolve_draws_vectorized(t1, t2, 1)[0]
            winner = t1 if resolved == 2 else t2
        loser = t2 if winner == t1 else t1
        
        qf_results[fixture["id"]] = {
            "id": fixture["id"],
            "date": fixture["date"],
            "stadium": fixture["stadium"],
            "t1": t1,
            "t2": t2,
            "winner": winner,
            "loser": loser,
            "probs": mc_res
        }
        
        match_records.append({
            "Stage": "Quarter-Finals",
            "Match_ID": fixture["id"],
            "Date": fixture["date"],
            "Team1": t1,
            "Team2": t2,
            "Prob_T1": (rf_t1 + xgb_t1 + ann_t1) / 3.0,
            "Prob_Draw": (rf_draw + xgb_draw + ann_draw) / 3.0,
            "Prob_T2": (rf_t2 + xgb_t2 + ann_t2) / 3.0,
            "RF_Prob_T1": rf_t1,
            "RF_Prob_Draw": rf_draw,
            "RF_Prob_T2": rf_t2,
            "XGB_Prob_T1": xgb_t1,
            "XGB_Prob_Draw": xgb_draw,
            "XGB_Prob_T2": xgb_t2,
            "ANN_Prob_T1": ann_t1,
            "ANN_Prob_Draw": ann_draw,
            "ANN_Prob_T2": ann_t2,
            "Winner": winner,
            "Loser": loser
        })
        
    # SF Simulation
    sf_results = {}
    sf_fixtures = [
        {"id": 101, "date": "Tuesday, 14 July 2026", "stadium": "Dallas Stadium", "t1_match": 97, "t2_match": 98},
        {"id": 102, "date": "Wednesday, 15 July 2026", "stadium": "Atlanta Stadium", "t1_match": 99, "t2_match": 100}
    ]
    
    for fixture in sf_fixtures:
        t1 = qf_results[fixture["t1_match"]]["winner"]
        t2 = qf_results[fixture["t2_match"]]["winner"]
        mc_res = run_monte_carlo_for_match(t1, t2)
        if raw_csv_path:
            save_raw_trials_to_csv(raw_csv_path, "Semi-Finals", fixture["id"], fixture["date"], t1, t2, mc_res)
        rf_t1, rf_draw, rf_t2 = mc_res["rf"]
        xgb_t1, xgb_draw, xgb_t2 = mc_res["xgb"]
        ann_t1, ann_draw, ann_t2 = mc_res["ann"]
        
        rf_pred = get_model_prediction(t1, t2, rf_t1, rf_draw, rf_t2, is_knockout=True)
        xgb_pred = get_model_prediction(t1, t2, xgb_t1, xgb_draw, xgb_t2, is_knockout=True)
        ann_pred = get_model_prediction(t1, t2, ann_t1, ann_draw, ann_t2, is_knockout=True)
        
        votes = [rf_pred, xgb_pred, ann_pred]
        if votes.count(2) >= 2:
            winner = t1
        elif votes.count(0) >= 2:
            winner = t2
        else:
            resolved = resolve_draws_vectorized(t1, t2, 1)[0]
            winner = t1 if resolved == 2 else t2
        loser = t2 if winner == t1 else t1
        
        sf_results[fixture["id"]] = {
            "id": fixture["id"],
            "date": fixture["date"],
            "stadium": fixture["stadium"],
            "t1": t1,
            "t2": t2,
            "winner": winner,
            "loser": loser,
            "probs": mc_res
        }
        
        match_records.append({
            "Stage": "Semi-Finals",
            "Match_ID": fixture["id"],
            "Date": fixture["date"],
            "Team1": t1,
            "Team2": t2,
            "Prob_T1": (rf_t1 + xgb_t1 + ann_t1) / 3.0,
            "Prob_Draw": (rf_draw + xgb_draw + ann_draw) / 3.0,
            "Prob_T2": (rf_t2 + xgb_t2 + ann_t2) / 3.0,
            "RF_Prob_T1": rf_t1,
            "RF_Prob_Draw": rf_draw,
            "RF_Prob_T2": rf_t2,
            "XGB_Prob_T1": xgb_t1,
            "XGB_Prob_Draw": xgb_draw,
            "XGB_Prob_T2": xgb_t2,
            "ANN_Prob_T1": ann_t1,
            "ANN_Prob_Draw": ann_draw,
            "ANN_Prob_T2": ann_t2,
            "Winner": winner,
            "Loser": loser
        })
        
    # Bronze Final
    t1_bronze = sf_results[101]["loser"]
    t2_bronze = sf_results[102]["loser"]
    mc_res = run_monte_carlo_for_match(t1_bronze, t2_bronze)
    if raw_csv_path:
        save_raw_trials_to_csv(raw_csv_path, "Bronze Final", 103, "Saturday, 18 July 2026", t1_bronze, t2_bronze, mc_res)
    rf_t1, rf_draw, rf_t2 = mc_res["rf"]
    xgb_t1, xgb_draw, xgb_t2 = mc_res["xgb"]
    ann_t1, ann_draw, ann_t2 = mc_res["ann"]
    
    rf_pred = get_model_prediction(t1_bronze, t2_bronze, rf_t1, rf_draw, rf_t2, is_knockout=True)
    xgb_pred = get_model_prediction(t1_bronze, t2_bronze, xgb_t1, xgb_draw, xgb_t2, is_knockout=True)
    ann_pred = get_model_prediction(t1_bronze, t2_bronze, ann_t1, ann_draw, ann_t2, is_knockout=True)
    
    votes = [rf_pred, xgb_pred, ann_pred]
    if votes.count(2) >= 2:
        winner_bronze = t1_bronze
    elif votes.count(0) >= 2:
        winner_bronze = t2_bronze
    else:
        resolved = resolve_draws_vectorized(t1_bronze, t2_bronze, 1)[0]
        winner_bronze = t1_bronze if resolved == 2 else t2_bronze
    loser_bronze = t2_bronze if winner_bronze == t1_bronze else t1_bronze
    
    bronze_result = {
        "id": 103,
        "date": "Saturday, 18 July 2026",
        "stadium": "Miami Stadium",
        "t1": t1_bronze,
        "t2": t2_bronze,
        "winner": winner_bronze,
        "loser": loser_bronze,
        "probs": mc_res
    }
    
    match_records.append({
        "Stage": "Bronze Final",
        "Match_ID": 103,
        "Date": "Saturday, 18 July 2026",
        "Team1": t1_bronze,
        "Team2": t2_bronze,
        "Prob_T1": (rf_t1 + xgb_t1 + ann_t1) / 3.0,
        "Prob_Draw": (rf_draw + xgb_draw + ann_draw) / 3.0,
        "Prob_T2": (rf_t2 + xgb_t2 + ann_t2) / 3.0,
        "RF_Prob_T1": rf_t1,
        "RF_Prob_Draw": rf_draw,
        "RF_Prob_T2": rf_t2,
        "XGB_Prob_T1": xgb_t1,
        "XGB_Prob_Draw": xgb_draw,
        "XGB_Prob_T2": xgb_t2,
        "ANN_Prob_T1": ann_t1,
        "ANN_Prob_Draw": ann_draw,
        "ANN_Prob_T2": ann_t2,
        "Winner": winner_bronze,
        "Loser": loser_bronze
    })
    
    # Grand Final
    t1_final = sf_results[101]["winner"]
    t2_final = sf_results[102]["winner"]
    mc_res = run_monte_carlo_for_match(t1_final, t2_final)
    if raw_csv_path:
        save_raw_trials_to_csv(raw_csv_path, "Grand Final", 104, "Sunday, 19 July 2026", t1_final, t2_final, mc_res)
    rf_t1, rf_draw, rf_t2 = mc_res["rf"]
    xgb_t1, xgb_draw, xgb_t2 = mc_res["xgb"]
    ann_t1, ann_draw, ann_t2 = mc_res["ann"]
    
    rf_pred = get_model_prediction(t1_final, t2_final, rf_t1, rf_draw, rf_t2, is_knockout=True)
    xgb_pred = get_model_prediction(t1_final, t2_final, xgb_t1, xgb_draw, xgb_t2, is_knockout=True)
    ann_pred = get_model_prediction(t1_final, t2_final, ann_t1, ann_draw, ann_t2, is_knockout=True)
    
    votes = [rf_pred, xgb_pred, ann_pred]
    if votes.count(2) >= 2:
        winner_final = t1_final
    elif votes.count(0) >= 2:
        winner_final = t2_final
    else:
        resolved = resolve_draws_vectorized(t1_final, t2_final, 1)[0]
        winner_final = t1_final if resolved == 2 else t2_final
    loser_final = t2_final if winner_final == t1_final else t1_final
    
    final_result = {
        "id": 104,
        "date": "Sunday, 19 July 2026",
        "stadium": "New York New Jersey Stadium",
        "t1": t1_final,
        "t2": t2_final,
        "winner": winner_final,
        "loser": loser_final,
        "probs": mc_res
    }
    
    match_records.append({
        "Stage": "Grand Final",
        "Match_ID": 104,
        "Date": "Sunday, 19 July 2026",
        "Team1": t1_final,
        "Team2": t2_final,
        "Prob_T1": (rf_t1 + xgb_t1 + ann_t1) / 3.0,
        "Prob_Draw": (rf_draw + xgb_draw + ann_draw) / 3.0,
        "Prob_T2": (rf_t2 + xgb_t2 + ann_t2) / 3.0,
        "RF_Prob_T1": rf_t1,
        "RF_Prob_Draw": rf_draw,
        "RF_Prob_T2": rf_t2,
        "XGB_Prob_T1": xgb_t1,
        "XGB_Prob_Draw": xgb_draw,
        "XGB_Prob_T2": xgb_t2,
        "ANN_Prob_T1": ann_t1,
        "ANN_Prob_Draw": ann_draw,
        "ANN_Prob_T2": ann_t2,
        "Winner": winner_final,
        "Loser": loser_final
    })
    
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
        "final_result": final_result,
        "match_records": match_records
    }

# Matches Raw from Streamlit app
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

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print("Starting Monte Carlo simulation for 2026 World Cup...")
    os.makedirs(os.path.join(project_root, "Data"), exist_ok=True)
    csv_path = os.path.join(project_root, "Data", "monte_carlo_results.csv")
    raw_csv_path = os.path.join(project_root, "Data", "monte_carlo_raw_trials.csv")
    
    if os.path.exists(raw_csv_path):
        try:
            os.remove(raw_csv_path)
        except Exception:
            pass
            
    sim_results = simulate_world_cup_monte_carlo(matches_raw, raw_csv_path=raw_csv_path)
    
    records_df = pd.DataFrame(sim_results["match_records"])
    records_df.to_csv(csv_path, index=False)
    
    print(f"Monte Carlo simulation completed!")
    print(f"Aggregated results saved to {csv_path}")
    print(f"Raw trial results saved to {raw_csv_path}")
    print(f"Total matches simulated: {len(records_df)}")
    print(f"World Cup Champion: {sim_results['final_result']['winner']}")

if __name__ == '__main__':
    main()