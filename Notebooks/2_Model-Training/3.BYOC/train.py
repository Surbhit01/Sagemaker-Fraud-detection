#!/usr/bin/env python3
"""SageMaker BYOC training entry point.

Copied into the image as /usr/local/bin/train (see Dockerfile) -- SageMaker
runs a training job with `docker run <image> train`, with no entry_point/
hyperparameter plumbing from the SDK. Everything here is read directly off
the filesystem paths SageMaker mounts into the container, per the BYOC
training contract:
  /opt/ml/input/data/<channel>/          -- input channels (train, validation)
  /opt/ml/input/config/hyperparameters.json -- hyperparameters (all values are strings)
  /opt/ml/model/                          -- write the trained model here
"""

import json
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

INPUT_DATA_PATH = "/opt/ml/input/data"
MODEL_PATH = "/opt/ml/model"
HYPERPARAM_PATH = "/opt/ml/input/config/hyperparameters.json"
TARGET_COL = "isFraud"


def load_channel(name: str):
    channel_dir = os.path.join(INPUT_DATA_PATH, name)
    files = [
        os.path.join(channel_dir, f)
        for f in os.listdir(channel_dir)
        if f.endswith(".csv")
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def main():
    with open(HYPERPARAM_PATH) as f:
        hp = json.load(f)  # every value arrives as a string -- cast explicitly

    X_train, y_train = load_channel("train")
    X_val, y_val = load_channel("validation")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(hp.get("max_depth", 5)),
        "eta": float(hp.get("eta", 0.2)),
        "scale_pos_weight": float(hp.get("scale_pos_weight", 1.0)),
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(hp.get("num_round", 100)),
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=10,
    )

    val_auc = roc_auc_score(y_val, booster.predict(dval))
    print(f"validation-auc: {val_auc:.4f}")

    booster.save_model(os.path.join(MODEL_PATH, "xgboost-model"))


if __name__ == "__main__":
    main()
