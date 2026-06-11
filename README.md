
## Project Structure


```
World_Cup_Project
├─ api.py
├─ app.py
├─ Data
│  ├─ cleaned
│  │  ├─ data_describe.csv
│  │  ├─ input_data.csv
│  │  ├─ teams_data.csv
│  │  └─ training_data.csv
│  └─ raw
│     └─ nations_data.csv
├─ Images
│  ├─ EDA
│  │  ├─ correlation_matrix.png
│  │  ├─ correlation_with_home_win.png
│  │  ├─ elo_diff_vs_outcome.png
│  │  ├─ elo_distribution.png
│  │  ├─ final_results_distribution.png
│  │  ├─ missing_values.png
│  │  ├─ outcomes_by_tournament.png
│  │  ├─ outcome_props_by_elo_quintiles.png
│  │  └─ performance_metrics_comparison.png
│  └─ World_Cup_Trophy
│     └─ World_Cup.png
├─ mlflow_script.py
├─ Notebooks
│  ├─ EDA.ipynb
│  ├─ Images
│  │  ├─ Model_Metrics
│  │  │  ├─ cross_val_score_of_plot.png
│  │  │  └─ test_set_performance.png
│  │  └─ Model_Performance
│  └─ model_metrics.ipynb
├─ README.md
├─ requirements.txt
└─ src
   ├─ ETL
   │  ├─ Extract
   │  │  ├─ scraper.py
   │  │  └─ __init__.py
   │  ├─ Load
   │  │  └─ data_inspection.ipynb
   │  ├─ Transform
   │  │  ├─ feature_engineering.py
   │  │  └─ __init__.py
   │  └─ __init__.py
   ├─ Models
   │  ├─ pipeline.py
   │  ├─ train.py
   │  └─ __init__.py
   └─ __init__.py

```