"""SageMaker script-mode training entry point.

Run by the managed XGBoost/SKLearn container via `entry_point="train.py"`.
SageMaker passes hyperparameters as CLI args and data/output locations as
environment variables (SM_CHANNEL_<NAME>, SM_MODEL_DIR) -- both are read
through argparse defaults below, which is the standard script-mode pattern.
"""

import argparse
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

TARGET_COL = "isFraud"


def parse_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters -- SageMaker passes these as --key value CLI args
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--scale_pos_weight", type=float, default=1.0)

    # Data/output locations -- SageMaker injects these as env vars
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    return parser.parse_args()


def load_channel(channel_dir: str):
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
    args = parse_args()

    X_train, y_train = load_channel(args.train)
    X_val, y_val = load_channel(args.validation)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "scale_pos_weight": args.scale_pos_weight,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=10,
    )

    val_auc = roc_auc_score(y_val, booster.predict(dval))
    print(f"validation-auc: {val_auc:.4f}")
    booster.save_model(os.path.join(args.model_dir, "xgboost-model"))
    print(f"Saving model to {args.model_dir}")



if __name__ == "__main__":
    main()
