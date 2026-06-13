import sys
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ETL.Transform.feature_engineering import preprocessing


def to_dense(X):
    """Utility to convert sparse matrices to dense arrays for robust model consumption."""
    if hasattr(X, "toarray"):
        return X.toarray()
    elif hasattr(X, "todense"):
        return X.todense()
    return X


def load_and_prepare_data():
    """Loads and preprocesses the football match dataset"""
    print("Loading raw nations dataset...")
    raw_data_path = os.path.join(project_root, "Data", "raw", "nations_data.csv")
    messy_dataset = pd.read_csv(raw_data_path)
    
    print("Preprocessing and engineering features...")
    dataset, team_stats, team_elos = preprocessing(messy_dataset)
    
    X = dataset.drop(columns=["Home Team", "Away Team", "Final Result"])
    y = dataset["Final Result"]
    
    target_mapping = {"A": 0, "D": 1, "H": 2}
    y = y.map(target_mapping)
    
    return X, y


def train_random_forest(X_tr, y_tr, cv_inner):
    """Grid searches and fits the best RandomForest model."""
    print("Tuning RandomForest Classifier...")
    rf = RandomForestClassifier(random_state=42)
    rf_grid = {
        "n_estimators": [40,55,60],
        "max_depth": [5,8,12],
        "criterion": ["gini", "entropy"]
    }
    grid = GridSearchCV(estimator=rf, param_grid=rf_grid, cv=cv_inner, scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    return grid.best_estimator_, grid.best_params_


def train_xgboost(X_tr, y_tr, cv_inner):
    """Grid searches XGBoost and returns calibrated estimator."""
    print("Tuning XGBoost Classifier...")
    xgb = XGBClassifier(random_state=42, eval_metric="mlogloss")
    xgb_grid = {
        "n_estimators": [42, 60],
        "max_depth": [3, 5],
        "learning_rate": [0.08, 0.15]
    }
    grid = GridSearchCV(estimator=xgb, param_grid=xgb_grid, cv=cv_inner, scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    best_xgb = grid.best_estimator_
    
    # Wrap in calibration to get reliable probability distributions
    print("Calibrating XGBoost Classifier...")
    calibrated_xgb = CalibratedClassifierCV(estimator=best_xgb, method="sigmoid")
    calibrated_xgb.fit(X_tr, y_tr)
    return calibrated_xgb, grid.best_params_


def train_logistic_regression(X_tr, y_tr, cv_inner):
    """Grid searches and fits the best LogisticRegression model."""
    print("Tuning Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr_grid = {
        "C": [0.01, 0.1, 1.0],
        "solver": ["saga"],
        "class_weight": [None, "balanced"]
    }
    grid = GridSearchCV(estimator=lr, param_grid=lr_grid, cv=cv_inner, scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    return grid.best_estimator_, grid.best_params_


def train_mlp_classifier(X_tr, y_tr, cv_inner):
    """Grid searches and fits the best MLPClassifier model."""
    print("Tuning MLP Neural Network Classifier...")
    mlp = MLPClassifier(validation_fraction=0.1,max_iter=1000, random_state=42)
    mlp_grid = {
        'hidden_layer_sizes': [(16,),(32,)],
        'alpha': [ 0.1, 1.0],
        'activation': ['relu','tanh'],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001]
    }
    grid = GridSearchCV(estimator=mlp, param_grid=mlp_grid, cv=cv_inner, scoring="neg_log_loss", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    model=grid.best_estimator_
    calibrated_model=CalibratedClassifierCV(estimator=model,method='sigmoid',n_jobs=-1,cv=TimeSeriesSplit(n_splits=5))
    calibrated_model.fit(X_tr,y_tr)
    return calibrated_model, grid.best_params_


def main():
    # Load dataset
    X, y = load_and_prepare_data()
    
    # 80/20 Split for final model training and validation pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    cv_outer = TimeSeriesSplit(n_splits=5)
    cv_inner = TimeSeriesSplit(n_splits=3)
    
    # Performance trackers
    model_performance = {
        "Random Forest": {"accuracies": [], "losses": [], "best_params": None},
        "XGBoost Calibrated": {"accuracies": [], "losses": [], "best_params": None},
        "Logistic Regression": {"accuracies": [], "losses": [], "best_params": None},
        "MLPClassifier": {"accuracies": [], "losses": [], "best_params": None}
    }
    
    categorical_features = ["tournament_name"]
    
    print("\nStarting TimeSeriesSplit Cross-Validation (5 Folds)...")
    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X)):
        print(f"\n--- Processing Fold {fold + 1} / 5 ---")
        
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        # Preprocessing setup (fit on train split only to prevent leakage)
        ct = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
            remainder="passthrough"
        )
        X_tr_trans = to_dense(ct.fit_transform(X_tr))
        X_te_trans = to_dense(ct.transform(X_te))
        
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_trans)
        X_te_imp = imputer.transform(X_te_trans)
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_te_scaled = scaler.transform(X_te_imp)
        
        rf_model, rf_params = train_random_forest(X_tr_imp, y_tr, cv_inner)
        rf_preds = rf_model.predict(X_te_imp)
        rf_probs = rf_model.predict_proba(X_te_imp)
        model_performance["Random Forest"]["accuracies"].append(accuracy_score(y_te, rf_preds))
        model_performance["Random Forest"]["losses"].append(log_loss(y_te, rf_probs, labels=[0, 1, 2]))
        model_performance["Random Forest"]["best_params"] = rf_params
        
        xgb_model, xgb_params = train_xgboost(X_tr_imp, y_tr, cv_inner)
        xgb_preds = xgb_model.predict(X_te_imp)
        xgb_probs = xgb_model.predict_proba(X_te_imp)
        model_performance["XGBoost Calibrated"]["accuracies"].append(accuracy_score(y_te, xgb_preds))
        model_performance["XGBoost Calibrated"]["losses"].append(log_loss(y_te, xgb_probs, labels=[0, 1, 2]))
        model_performance["XGBoost Calibrated"]["best_params"] = xgb_params
        
        lr_model, lr_params = train_logistic_regression(X_tr_scaled, y_tr, cv_inner)
        lr_preds = lr_model.predict(X_te_scaled)
        lr_probs = lr_model.predict_proba(X_te_scaled)
        model_performance["Logistic Regression"]["accuracies"].append(accuracy_score(y_te, lr_preds))
        model_performance["Logistic Regression"]["losses"].append(log_loss(y_te, lr_probs, labels=[0, 1, 2]))
        model_performance["Logistic Regression"]["best_params"] = lr_params
        
        mlp_model, mlp_params = train_mlp_classifier(X_tr_scaled, y_tr, cv_inner)
        mlp_preds = mlp_model.predict(X_te_scaled)
        mlp_probs = mlp_model.predict_proba(X_te_scaled)
        model_performance["MLPClassifier"]["accuracies"].append(accuracy_score(y_te, mlp_preds))
        model_performance["MLPClassifier"]["losses"].append(log_loss(y_te, mlp_probs, labels=[0, 1, 2]))
        model_performance["MLPClassifier"]["best_params"] = mlp_params
        
    print("\nTimeSeriesSplit Cross-Validation Complete.")
    
    print("\nFitting final preprocessing pipelines on training set...")
    ct_final = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough"
    )
    X_train_trans = to_dense(ct_final.fit_transform(X_train))
    
    imputer_final = SimpleImputer(strategy="median")
    X_train_imp = imputer_final.fit_transform(X_train_trans)
    
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_imp)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Football_Match_Prediction_Pipeline")
    
    with mlflow.start_run(run_name="TimeSeriesCV_Outer_Pipeline") as parent_run:
        mlflow.log_param("outer_cv_splits", 5)
        mlflow.log_param("inner_cv_splits", 3)
        
        best_mean_loss = float("inf")
        champion_run_id = None
        champion_model_name = None
        
        for model_name, perf in model_performance.items():
            mean_acc = np.mean(perf["accuracies"])
            std_acc = np.std(perf["accuracies"])
            mean_loss = np.mean(perf["losses"])
            std_loss = np.std(perf["losses"])
            
            print(f"\n{model_name} Cross-Validation Performance:")
            print(f"  Mean Accuracy: {mean_acc:.4%} (±{std_acc:.4%})")
            print(f"  Mean Log Loss: {mean_loss:.4f} (±{std_loss:.4f})")
            
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                for param_name, param_val in perf["best_params"].items():
                    mlflow.log_param(f"best_{param_name}", param_val)
                
                mlflow.log_metric("mean_accuracy", mean_acc)
                mlflow.log_metric("std_accuracy", std_acc)
                mlflow.log_metric("mean_log_loss", mean_loss)
                mlflow.log_metric("std_log_loss", std_loss)
                
                print(f"Training final {model_name} model on full training dataset...")
                if model_name == "Random Forest":
                    model = RandomForestClassifier(**perf["best_params"], random_state=42)
                    model.fit(X_train_imp, y_train)
                    mlflow.sklearn.log_model(model, "model")
                elif model_name == "XGBoost Calibrated":
                    xgb_clf = XGBClassifier(**perf["best_params"], random_state=42, eval_metric="mlogloss")
                    model = CalibratedClassifierCV(estimator=xgb_clf, method="sigmoid")
                    model.fit(X_train_imp, y_train)
                    mlflow.sklearn.log_model(model, "model")
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(**perf["best_params"], max_iter=2000, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    mlflow.sklearn.log_model(model, "model")
                elif model_name == "MLPClassifier":
                    model = MLPClassifier(**perf["best_params"], max_iter=1000, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    mlflow.sklearn.log_model(model, "model")
                
                if mean_loss < best_mean_loss:
                    best_mean_loss = mean_loss
                    champion_run_id = child_run.info.run_id
                    champion_model_name = model_name
                    
        
        print(f"Best Model: {champion_model_name} (Mean Log Loss: {best_mean_loss:.4f})")
        
        registry_name = "Football_Match_Predictor"
        model_uri = f"runs:/{champion_run_id}/model"
        
        print(f"Registering best model under name '{registry_name}'...")
        model_details = mlflow.register_model(model_uri=model_uri, name=registry_name)
        
        print(f"Registered model version: {model_details.version}")
        
        client = mlflow.tracking.MlflowClient()

        print("Setting model alias 'champion' for the FastAPI production environment...")
        client.set_registered_model_alias(
            name=registry_name,
            alias="champion",
            version=model_details.version
        )

        print("Tagging model version with stage='Production'...")
        client.set_model_version_tag(
            name=registry_name,
            version=model_details.version,
            key="stage",
            value="Production"
        )
        
        print("Transitioning model stage to Production...")
        client.transition_model_version_stage(
            name=registry_name,
            version=model_details.version,
            stage="Production"
        )
        
        print(f"\nSuccessfully logged all runs. The best model '{champion_model_name}' registered as version {model_details.version} in Production")


if __name__ == "__main__":
    main()
