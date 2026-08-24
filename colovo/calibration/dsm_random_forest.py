#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from configs.config import RF_ESTIMATORS, RF_RANDOM_STATE, DSM_MIN, DSM_MAX
from colovo.utils.paths import display_path

FEATURE_COLUMNS = ["hue_median",
                   "sat_median",
                   "val_median",
                   "lab_a_median",
                   "lab_b_median"]

def train_dsm_model(features_df, model_path):
    X = features_df[FEATURE_COLUMNS].values
    y = features_df["DSM"].values

    model = RandomForestRegressor(n_estimators = RF_ESTIMATORS, random_state = RF_RANDOM_STATE)
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok = True)

    joblib.dump(model, model_path)

    print(f"[INFO] Modelo salvo em: {display_path(model_path, 'models')}")

def load_dsm_model(path):
    return joblib.load(path)

def predict_dsm(model, features):
    pred = model.predict([features])[0]
    pred = max(DSM_MIN, min(DSM_MAX, pred))
    return float(pred)
