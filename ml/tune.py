"""Hyperparameter tuning for the winning classifier with Optuna.

Responsible for: running an Optuna study over the model that won in train.py,
searching its hyperparameters (and the TF-IDF settings) to squeeze out extra
macro-F1, then saving the tuned model if it actually improves on the default.

Inputs: the labeled training data, the winning model type from train.py, a
trial budget of about 50 trials, and the macro-F1 objective.

Outputs: an Optuna study with per-trial scores, the best parameter set printed
and saved, a tuned .joblib in models/, and a model_runs row for the tuned model.

How I'll know it works: the study completes 50 trials without error, the best
trial's macro-F1 beats or ties the default model's score from train.py, and the
tuned artifact loads and predicts.
"""
import sys
import json
import datetime
from pathlib import Path
import joblib
import numpy as np
import optuna
import torch
import sqlite3
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import SubFigure
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



conn = sqlite3.connect(config.DB_PATH)
df = pd.read_sql_query("SELECT p.title, p.description, l.category "
                        "FROM postings p "
                        "JOIN labels l ON l.posting_id = p.id", conn)

X = df["title"] + " "+ df["description"]
Y = df["category"]

# Same split as train.py, so the tuned model is tested on the same postings.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify= Y, test_size= 0.2, random_state= config.RANDOM_SEED)

# The vectorizer is NOT loaded: its settings are tuned, so each trial fits its own.
encoder = joblib.load(config.LABEL_ENCODER_PATH)
y_train = encoder.transform(Y_train)
y_test = encoder.transform(Y_test)
n_classes = len(encoder.classes_)


def suggest_vectorizer(trial):
    """Build a TF-IDF vectorizer from this trial's choices."""
    ngram_max = trial.suggest_int("ngram_max", 1, 2)
    min_df = trial.suggest_int("min_df", 1, 5)
    max_features = trial.suggest_int("max_features", 5000, 30000, step=5000)
    return TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, max_features=max_features)


def define_model(trial, in_features):
    """Build the MLP: (Linear -> ReLU -> Dropout) x n_layers, then the output layer."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    layers = []
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 32, 512, log=True)
        layers += [nn.Linear(in_features, out_features), nn.ReLU(), nn.Dropout(dropout)]
        in_features = out_features
    layers.append(nn.Linear(in_features, n_classes))
    return nn.Sequential(*layers)


def fit_and_score(model, lr, n_epoch, x_fit, y_fit, x_eval, y_eval):
    """Train on (x_fit, y_fit), return macro-F1 on (x_eval, y_eval).

    x_* are sparse TF-IDF matrices; each batch is made dense on its own so the
    whole matrix never has to fit in memory as a dense tensor.
    """
    torch.manual_seed(config.RANDOM_SEED)
    rng = np.random.default_rng(config.RANDOM_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    y_fit_t = torch.tensor(y_fit, dtype=torch.long)

    model.train()
    for _ in range(n_epoch):
        order = rng.permutation(x_fit.shape[0])
        for start in range(0, len(order), config.BATCH_SIZE):
            idx = order[start:start + config.BATCH_SIZE]
            x_batch = torch.tensor(x_fit[idx].toarray(), dtype=torch.float32)
            loss = criterion(model(x_batch), y_fit_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(x_eval.toarray(), dtype=torch.float32)).argmax(1).numpy()
    return f1_score(y_eval, preds, average="macro")


def objective(trial):
    """Mean macro-F1 over stratified CV folds of the TRAINING set only."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    n_epoch = trial.suggest_int("n_epoch", 5, 30)
    skf = StratifiedKFold(n_splits=config.TUNING_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    scores = []
    for fold, (fit_idx, eval_idx) in enumerate(skf.split(X_train, y_train)):
        # Fit TF-IDF inside the fold so it never sees the held-out text.
        vectorizer = suggest_vectorizer(trial)
        x_fit = vectorizer.fit_transform(X_train.iloc[fit_idx])
        x_eval = vectorizer.transform(X_train.iloc[eval_idx])
        model = define_model(trial, x_fit.shape[1])
        scores.append(fit_and_score(model, lr, n_epoch, x_fit, y_train[fit_idx], x_eval, y_train[eval_idx]))

        # Report the running mean so the pruner can stop clearly bad trials early.
        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
study.optimize(objective, n_trials=config.TUNING_N_TRIALS)

print(f"Best CV macro-F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# Retrain the best setting on the full training set, score once on the test set.
best = optuna.trial.FixedTrial(study.best_params)
best_vectorizer = suggest_vectorizer(best)
x_train_full = best_vectorizer.fit_transform(X_train)
x_test_full = best_vectorizer.transform(X_test)
best_model = define_model(best, x_train_full.shape[1])
tuned_f1 = fit_and_score(best_model, study.best_params["lr"], study.best_params["n_epoch"],
                         x_train_full, y_train, x_test_full, y_test)

# Default score = the latest MLP row train.py wrote.
cur = conn.cursor()
default_f1 = cur.execute("SELECT macro_f1 FROM model_runs WHERE model_name = ? ORDER BY id DESC LIMIT 1",
                         (config.MODEL_NAME_MLP,)).fetchone()[0]
gain = tuned_f1 - default_f1
print(f"Default MLP test macro-F1: {default_f1:.4f}")
print(f"Tuned   MLP test macro-F1: {tuned_f1:.4f}")


cur.execute('INSERT INTO model_runs(run_date, model_name, macro_f1, notes) VALUES(?,?,?,?)',
            (datetime.date.today().isoformat(), config.MODEL_NAME_MLP_TUNED, tuned_f1,
             f"Optuna {config.TUNING_N_TRIALS} trials, {config.TUNING_CV_FOLDS}-fold CV. "
             f"Best params: {json.dumps(study.best_params)}"))
conn.commit()

