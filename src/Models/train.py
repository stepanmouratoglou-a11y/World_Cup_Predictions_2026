from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit,GridSearchCV
from sklearn.neural_network import MLPClassifier


def xgboost_model(X_train,y_train):
    tscv=TimeSeriesSplit(n_splits=5)
    xgb=XGBClassifier(random_state=42,eval_metric='mlogloss')
    parameters={
        'n_estimators':[42,67,80],
        'max_depth':[3,5,6],
        'learning_rate':[0.08,0.1,0.15,0.22]
    }

    grid_search=GridSearchCV(estimator=xgb,
                             param_grid=parameters,
                             n_jobs=-1,
                             scoring='neg_log_loss',
                             cv=tscv)
    grid_search.fit(X_train,y_train)
    model=grid_search.best_estimator_

    classifier=CalibratedClassifierCV(estimator=model,method='sigmoid')
    classifier.fit(X_train,y_train)
    
    return classifier

def random_forest_classifier(X_train,y_train):
    tscv=TimeSeriesSplit(n_splits=5)

    classifier=RandomForestClassifier(random_state=42)
    
    parameters={
        "n_estimators": [40,55,60],
        "max_depth": [5,8,12],
        "criterion": ["gini"],
        "min_samples_split": [2, 5, 10]
    }

    grid_search=GridSearchCV(
        param_grid=parameters,
        cv=tscv,
        estimator=classifier,
        n_jobs=-1,
        scoring='neg_log_loss'
    )

    grid_search.fit(X_train,y_train)

    model=grid_search.best_estimator_

    return model

def logistic_regression(X_train,y_train):
    tscv=TimeSeriesSplit(n_splits=5)
    classifier=LogisticRegression(max_iter=1900,random_state=42)

    parameters={
        'C': [0.01, 0.1, 1],
        'solver': ['lbfgs', 'saga'],
        'class_weight': [None, 'balanced']
    }

    grid_search=GridSearchCV(
        estimator=classifier,
        param_grid=parameters,
        n_jobs=5,
        scoring='neg_log_loss',
        cv=tscv
    )

    grid_search.fit(X_train,y_train)

    model=grid_search.best_estimator_

    return model



def artificial_neural_network(X_train,y_train):
    tscv=TimeSeriesSplit(n_splits=5)

    classifier=MLPClassifier(
        early_stopping=True, 
        validation_fraction=0.1, 
        max_iter=1970, 
        random_state=42
    )

    parameters = {
        'hidden_layer_sizes': [(128,64),(64,32)],
        'alpha': [1e-5, 0.001],
        'activation': ['relu'],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001]
    }

    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=parameters,
        n_jobs=-1,
        scoring='neg_log_loss',
        cv=tscv
    )

    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    return model