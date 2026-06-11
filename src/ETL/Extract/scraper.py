import time
import pandas as pd
from playwright.sync_api import sync_playwright
import os
from pathlib import Path

# Verified Sofascore Mapping Dictionary for  45 teams
TARGET_TEAMS = {
    "Algeria": 4743,
    "Argentina": 4819,
    "Australia": 4767,
    "Austria": 4716,
    "Belgium": 4715,
    "Bosnia and Herzegovina": 4479,
    "Brazil": 4748,
    "Cabo Verde": 4753,
    "Colombia": 4820,
    "DR Congo": 4823,
    "Côte d'Ivoire": 4718,
    "Croatia": 4705,
    "Curaçao": 55827,
    "Czechia": 4706,
    "Ecuador": 4734,
    "Egypt": 4758,
    "England": 4713,
    "France": 4481,
    "Germany": 4711,
    "Ghana": 4742,
    "Haiti": 7229,
    "Iran": 4766,
    "Iraq": 4751,
    "Japan": 4768,
    "Jordan": 4782,
    "South Korea": 4757,
    "Morocco": 4745,
    "Netherlands": 4712,
    "New Zealand": 4772,
    "Norway": 4702,
    "Panama": 4735,
    "Paraguay": 4731,
    "Portugal": 4704,
    "Qatar": 4762,
    "Saudi Arabia": 4761,
    "Scotland": 4714,
    "Senegal": 4741,
    "South Africa": 4739,
    "Spain": 4774,
    "Sweden": 4703,
    "Switzerland": 4720,
    "Tunisia": 4746,
    "Türkiye": 4709,
    "Uruguay": 4725,
    "Uzbekistan": 4723
}


def build_world_cup_dataset(target_games=40):
    master_dataset = []
    all_collected_ids = set()

    print("Launching browser context...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # Authorize the connection footprint
        page.goto("https://www.sofascore.com", wait_until="domcontentloaded")
        time.sleep(3)

        for country_name, team_id in TARGET_TEAMS.items():
            print(f"\nProcessing match logs for: {country_name}")
            match_metadata = []
            page_num = 0

            while len(match_metadata) < target_games:
                url = f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/{page_num}"
                api_response = page.evaluate(f"""
                    async () => {{
                        try {{
                            const res = await fetch('{url}');
                            if (res.status !== 200) return null;
                            return await res.json();
                        }} catch (e) {{ return null; }}
                    }}
                """)

                if not api_response or not api_response.get('events'):
                    break

                for event in api_response.get('events', []):
                    if event.get('status', {}).get('type') == 'finished':
                        # Capture essential context tags
                        m_id = event['id']
                        h_name = event.get('homeTeam', {}).get('name', 'Unknown')
                        a_name = event.get('awayTeam', {}).get('name', 'Unknown')

                        # Added metadata features
                        h_score = event.get('homeScore', {}).get('current', 0)
                        a_score = event.get('awayScore', {}).get('current', 0)
                        m_timestamp = event.get('startTimestamp', 0)
                        tourney_name = event.get('tournament', {}).get('name', 'Unknown Tournament')
                        is_neutral = 1 if event.get('neutralGround', False) else 0

                        match_metadata.append(
                            (m_id, h_name, a_name, h_score, a_score, m_timestamp, tourney_name, is_neutral))

                page_num += 1
                time.sleep(1.2)

            match_metadata = match_metadata[:target_games]
            print(f"  -> Extracted {len(match_metadata)} target match references.")

            # Deep scrape metric features
            for m_id, home_team, away_team, home_score, away_score, m_timestamp, tourney_name, is_neutral in match_metadata:
                if m_id in all_collected_ids:
                    continue

                stats_url = f"https://api.sofascore.com/api/v1/event/{m_id}/statistics"
                stats_response = page.evaluate(f"""
                    async () => {{
                        try {{
                            const res = await fetch('{stats_url}');
                            if (res.status !== 200) return null;
                            return await res.json();
                        }} catch (e) {{ return null; }}
                    }}
                """)

                if not stats_response:
                    continue

                # Merging basic context features + target score variables directly into the flat dict
                flat_stats = {
                    "match_id": m_id,
                    "match_timestamp": m_timestamp,
                    "tournament_name": tourney_name,
                    "is_neutral_ground": is_neutral,
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "home_final_score": home_score,
                    "away_final_score": away_score
                }

                # Append granular statistics items
                for period_data in stats_response.get("statistics", []):
                    if period_data.get("period") == "ALL":
                        for group in period_data.get("groups", []):
                            for item in group.get("statisticsItems", []):
                                metric = item["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")

                                h_val = item["home"].replace("%", "") if isinstance(item["home"], str) else item["home"]
                                a_val = item["away"].replace("%", "") if isinstance(item["away"], str) else item["away"]

                                try:
                                    flat_stats[f"home_{metric}"] = float(h_val) if '.' in str(h_val) else int(h_val)
                                    flat_stats[f"away_{metric}"] = float(a_val) if '.' in str(a_val) else int(a_val)
                                except ValueError:
                                    flat_stats[f"home_{metric}"] = h_val
                                    flat_stats[f"away_{metric}"] = a_val

                # Validation length change due to our newly appended columns
                if len(flat_stats) > 8:
                    master_dataset.append(flat_stats)
                    all_collected_ids.add(m_id)

                time.sleep(1.0)

        browser.close()
    return master_dataset


if __name__ == "__main__":
    dataset_list = build_world_cup_dataset(target_games=40)

    if dataset_list:
        df = pd.DataFrame(dataset_list)
        df['match_date'] = pd.to_datetime(df['match_timestamp'], unit='s')
        df = df.drop(columns=['match_timestamp'])
        df = df.sort_values(by='match_date', ascending=True)
        cols = ['match_id', 'match_date', 'tournament_name', 'is_neutral_ground', 'Home Team', 'Away Team',
                'home_final_score', 'away_final_score']
        remaining_cols = [c for c in df.columns if c not in cols]
        df = df[cols + remaining_cols]
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        output_dir = BASE_DIR / "Data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "nations_data.csv"
        df.to_csv(file_path, index=False)
        print(f"\nCompleted extraction! Saved clean table chronologically to '{file_path}'")
    else:
        print("\nNo items extracted. Check connection filters or terminal scripts.")