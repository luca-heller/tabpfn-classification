import os
import argparse
import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def parse_args():
    parser = argparse.ArgumentParser(description="TabPFN classification test")

    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--sampling", choices=["stratified", "random"], default="stratified")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Prepare data
    df_subset = df[args.features + [args.target]].dropna()
    X = df_subset[args.features].astype(np.float32)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df_subset[args.target])

    # Stratified or random split
    if args.sampling == "stratified":
        classes = np.unique(y)
        splitter = StratifiedShuffleSplit(n_splits=10, train_size=args.train_size, random_state=args.seed)
        for train_idx, test_idx in splitter.split(X, y):
            if set(np.unique(y[train_idx])) == set(classes):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                break
        else:
            raise ValueError("Stratified split failed to include all classes in training set.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.train_size, random_state=args.seed, shuffle=True
        )

    # Save splits
    pd.concat([X_train, pd.Series(y_train, name=args.target)], axis=1).to_csv(
        os.path.join(args.output_dir, "train_data.csv"), index=False
    )
    pd.concat([X_test, pd.Series(y_test, name=args.target)], axis=1).to_csv(
        os.path.join(args.output_dir, "test_data.csv"), index=False
    )
    print("Saved train/test splits.")

    # Base TabPFN model
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device=device,
        ignore_pretraining_limits=True,
        inference_config={"SUBSAMPLE_SAMPLES": 10000},
    )

    # Train
    print("Training model...")
    start_train = time.time()
    model.fit(X_train.to_numpy(), y_train)
    train_time = time.time() - start_train
    print(f"Training finished in {train_time:.2f} sec")

    # Predict in batches
    start_pred = time.time()
    all_probs = []
    for start in range(0, len(X_test), args.batch_size):
        batch = X_test.iloc[start:start + args.batch_size].to_numpy()
        probs = model.predict_proba(batch)
        all_probs.append(probs)

    prediction_probabilities = np.vstack(all_probs)
    pred_time = time.time() - start_pred

    y_pred = np.argmax(prediction_probabilities, axis=1)
    y_true = y_test

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    try:
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, prediction_probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, prediction_probabilities, multi_class="ovr", average="weighted")
    except:
        roc_auc = None

    print("\n RESULTS: ")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"ROC AUC:    {roc_auc:.4f}" if roc_auc is not None else "ROC AUC:    N/A")
    print(f"Train time: {train_time:.2f} sec")
    print(f"Pred time:  {pred_time:.2f} sec")

    # Save results
    out = {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "train_time": train_time,
        "pred_time": pred_time,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    pd.DataFrame([out]).to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print(f"Saved results: {args.output_dir}/results.csv")


if __name__ == "__main__":
    main()
