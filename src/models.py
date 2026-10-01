import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

from config import (
    PROCESSED_DIR,
    TRAIN_FRAC,
    VAL_FRAC,
    CV_N_FOLDS,
    STABLE_THRESHOLD,
    CATEGORY_STABLE_THRESHOLDS,
    USE_CATEGORY_THRESHOLDS,
    LIGHTGBM_DETERMINISTIC,
)

_DETERMINISM_PARAMS = {"deterministic": True, "force_row_wise": True} if LIGHTGBM_DETERMINISTIC else {}

REGRESSION_PARAMS = {"objective": "regression", "metric": "mae", "verbosity": -1, "seed": 42, **_DETERMINISM_PARAMS}
CLASSIFIER_PARAMS = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1, "seed": 42, **_DETERMINISM_PARAMS}

FEATURE_COLS = [
    "month",
    "is_summer",
    "is_winter",
    "is_near_black_friday",
    "is_near_midyear_sale",
    "is_near_holiday_season",
    "ratio_lag_1",
    "ratio_lag_4",
    "ratio_lag_8",
    "ratio_rolling_4",
    "ratio_rolling_8",
    "trend_slope",
    "avg_temp",
    "temp_anomaly",
    "precipitation",
    "precip_anomaly",
]

DIRECTION_LABELS = ("fall", "stable", "rise")  # fixed class order used everywhere a probability vector is involved


def get_category_thresholds(df: pd.DataFrame, category_col: str = "garment_group_name") -> np.ndarray:
    """Per-row stable-threshold array. When USE_CATEGORY_THRESHOLDS is True,
    looks up each row's category in CATEGORY_STABLE_THRESHOLDS (falling back
    to the pooled STABLE_THRESHOLD for any category missing from that dict).
    When False (the current default — see config.py for why this was tested
    and reverted), every row just gets the pooled STABLE_THRESHOLD, so the
    pipeline behaves exactly as it did before per-category thresholds were
    introduced. direction_from_pct_change/directional_accuracy both accept
    an array here via normal numpy broadcasting, so callers don't need to
    know which mode is active."""
    if not USE_CATEGORY_THRESHOLDS:
        return np.full(len(df), STABLE_THRESHOLD)
    return df[category_col].map(CATEGORY_STABLE_THRESHOLDS).fillna(STABLE_THRESHOLD).values


def chronological_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
    """Per-category chronological (non-shuffled) train/val/test split, to
    avoid time-series leakage. Categories are read dynamically from the
    dataframe rather than hardcoded."""
    train_parts, val_parts, test_parts = [], [], []
    for category, group in df.groupby("garment_group_name"):
        group = group.sort_values("week_start").reset_index(drop=True)
        n = len(group)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])
    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


def time_series_cv_splits(df: pd.DataFrame, n_folds: int = CV_N_FOLDS):
    """Expanding-window CV folds, computed per category then combined, so a
    fold's train indices never include a later week than its val indices for
    any category."""
    folds = [[] for _ in range(n_folds)]
    for category, group in df.groupby("garment_group_name"):
        group = group.sort_values("week_start")
        idx = group.index.to_numpy()
        n = len(idx)
        fold_size = n // (n_folds + 1)
        if fold_size < 1:
            continue
        for fold_i in range(n_folds):
            train_end = fold_size * (fold_i + 1)
            val_end = fold_size * (fold_i + 2) if fold_i < n_folds - 1 else n
            train_idx = idx[:train_end]
            val_idx = idx[train_end:val_end]
            folds[fold_i].append((train_idx, val_idx))
    combined = []
    for fold_parts in folds:
        train_idx = np.concatenate([t for t, v in fold_parts])
        val_idx = np.concatenate([v for t, v in fold_parts])
        combined.append((train_idx, val_idx))
    return combined


def naive_persistence_forecast(df: pd.DataFrame) -> np.ndarray:
    """Baseline: predict target=0 (no change)."""
    return np.zeros(len(df))


def direction_from_pct_change(pct_change: np.ndarray, threshold=STABLE_THRESHOLD) -> np.ndarray:
    """threshold may be a scalar (applied to every row) or an array the same
    length as pct_change (e.g. from get_category_thresholds), via ordinary
    numpy broadcasting."""
    direction = np.where(pct_change > threshold, "rise", np.where(pct_change < -threshold, "fall", "stable"))
    return direction


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold=STABLE_THRESHOLD) -> float:
    true_dir = direction_from_pct_change(y_true, threshold)
    pred_dir = direction_from_pct_change(y_pred, threshold)
    return (true_dir == pred_dir).mean()


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, threshold=STABLE_THRESHOLD):
    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    acc = directional_accuracy(y_true, y_pred, threshold)
    print(f"[{name}] MAE={mae:.4f}  RMSE={rmse:.4f}  Directional Accuracy={acc * 100:.2f}%")
    return {"name": name, "mae": mae, "rmse": rmse, "directional_accuracy": acc}


def evaluate_classifier(name: str, y_true_dir: np.ndarray, y_pred_dir: np.ndarray):
    acc = (y_true_dir == y_pred_dir).mean()
    print(f"[{name}] Directional Accuracy={acc * 100:.2f}%")
    return {"name": name, "directional_accuracy": acc}


def ensemble_direction(*direction_arrays) -> np.ndarray:
    """Unweighted majority vote across direction predictions. Last array wins
    ties (intended to be called with the classifier's predictions last)."""
    stacked = np.stack(direction_arrays, axis=0)
    out = []
    for col in stacked.T:
        vals, counts = np.unique(col, return_counts=True)
        max_count = counts.max()
        winners = vals[counts == max_count]
        if len(winners) == 1:
            out.append(winners[0])
        else:
            out.append(col[-1])  # tie -> last model's vote (classifier)
    return np.array(out)


def weighted_ensemble_direction(
    hard_votes: list,
    hard_weights: list,
    classifier_probs: np.ndarray = None,
    classifier_weight: float = 0.0,
    label_order=DIRECTION_LABELS,
) -> np.ndarray:
    """Confidence-weighted vote, tried as a refinement over the plain
    majority-vote ensemble above: each hard-direction model's vote is scaled
    by its own CV-measured directional accuracy, and the classifier
    contributes its full probability distribution rather than just its
    argmax. NOT used in the main pipeline below — kept here only for
    reference/reproducibility, since testing showed it slightly
    underperformed the plain ensemble on live Trends data (39.52% vs
    40.95%), most likely because CV-accuracy estimates from only 3 folds
    over a small dataset are themselves too noisy to reliably rank three
    already-similar models. A documented negative result, not a bug.
    """
    label_order = list(label_order)
    n = len(hard_votes[0])
    tally = np.zeros((n, len(label_order)))

    for votes, weight in zip(hard_votes, hard_weights):
        for i, direction in enumerate(votes):
            tally[i, label_order.index(direction)] += weight

    if classifier_probs is not None and classifier_weight > 0:
        tally += classifier_weight * classifier_probs

    winners = [label_order[i] for i in np.argmax(tally, axis=1)]
    return np.array(winners)


def train_lightgbm_cv(train_val_df: pd.DataFrame, feature_cols=FEATURE_COLS, target_col: str = "target"):
    folds = time_series_cv_splits(train_val_df)
    best_iterations = []
    fold_dir_accs = []
    for train_idx, val_idx in folds:
        train_data = lgb.Dataset(train_val_df.loc[train_idx, feature_cols], label=train_val_df.loc[train_idx, target_col])
        val_data = lgb.Dataset(train_val_df.loc[val_idx, feature_cols], label=train_val_df.loc[val_idx, target_col])
        model = lgb.train(
            REGRESSION_PARAMS,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        best_iterations.append(model.best_iteration)
        val_df = train_val_df.loc[val_idx]
        val_preds = model.predict(val_df[feature_cols], num_iteration=model.best_iteration)
        fold_dir_accs.append(directional_accuracy(val_df[target_col].values, val_preds, threshold=get_category_thresholds(val_df)))
    n_estimators = max(1, int(np.mean(best_iterations)))
    cv_dir_acc = float(np.mean(fold_dir_accs))
    print(f"LightGBM CV: fold best_iterations={best_iterations} -> using n_estimators={n_estimators} "
          f"(mean CV directional accuracy={cv_dir_acc * 100:.2f}%)")

    final_data = lgb.Dataset(train_val_df[feature_cols], label=train_val_df[target_col])
    final_model = lgb.train(
        REGRESSION_PARAMS,
        final_data,
        num_boost_round=n_estimators,
    )
    return final_model, cv_dir_acc


def train_ridge_cv(train_val_df: pd.DataFrame, feature_cols=FEATURE_COLS, target_col: str = "target"):
    folds = time_series_cv_splits(train_val_df)
    alphas = [0.1, 1.0, 10.0, 100.0]
    mean_maes = []
    mean_dir_accs = []
    for alpha in alphas:
        fold_maes = []
        fold_dir_accs = []
        for train_idx, val_idx in folds:
            model = Ridge(alpha=alpha)
            model.fit(train_val_df.loc[train_idx, feature_cols], train_val_df.loc[train_idx, target_col])
            val_df = train_val_df.loc[val_idx]
            preds = model.predict(val_df[feature_cols])
            fold_maes.append(np.abs(val_df[target_col] - preds).mean())
            fold_dir_accs.append(directional_accuracy(val_df[target_col].values, preds, threshold=get_category_thresholds(val_df)))
        mean_maes.append(np.mean(fold_maes))
        mean_dir_accs.append(np.mean(fold_dir_accs))
    best_i = int(np.argmin(mean_maes))
    best_alpha = alphas[best_i]
    cv_dir_acc = float(mean_dir_accs[best_i])
    print(f"Ridge CV: selected alpha={best_alpha} (mean CV MAE={mean_maes[best_i]:.4f}, "
          f"mean CV directional accuracy={cv_dir_acc * 100:.2f}%)")

    final_model = Ridge(alpha=best_alpha)
    final_model.fit(train_val_df[feature_cols], train_val_df[target_col])
    return final_model, cv_dir_acc


def train_direction_classifier_cv(train_val_df: pd.DataFrame, feature_cols=FEATURE_COLS, target_col: str = "target"):
    """Trains directly on the 3-way direction label (loss aligned with the
    evaluation metric), rather than thresholding a regressor's output."""
    labels = direction_from_pct_change(train_val_df[target_col].values, threshold=get_category_thresholds(train_val_df))
    label_map = {"fall": 0, "stable": 1, "rise": 2}
    y = np.array([label_map[l] for l in labels])

    folds = time_series_cv_splits(train_val_df)
    best_iterations = []
    fold_dir_accs = []
    for train_idx, val_idx in folds:
        train_data = lgb.Dataset(train_val_df.loc[train_idx, feature_cols], label=y[train_val_df.index.get_indexer(train_idx)])
        val_data = lgb.Dataset(train_val_df.loc[val_idx, feature_cols], label=y[train_val_df.index.get_indexer(val_idx)])
        model = lgb.train(
            CLASSIFIER_PARAMS,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        best_iterations.append(model.best_iteration)
        val_y_true = y[train_val_df.index.get_indexer(val_idx)]
        val_probs = model.predict(train_val_df.loc[val_idx, feature_cols], num_iteration=model.best_iteration)
        val_pred_idx = np.argmax(val_probs, axis=1)
        fold_dir_accs.append((val_y_true == val_pred_idx).mean())
    n_estimators = max(1, int(np.mean(best_iterations)))
    cv_dir_acc = float(np.mean(fold_dir_accs))
    print(f"Direction Classifier CV: fold best_iterations={best_iterations} -> using n_estimators={n_estimators} "
          f"(mean CV directional accuracy={cv_dir_acc * 100:.2f}%)")

    final_data = lgb.Dataset(train_val_df[feature_cols], label=y)
    final_model = lgb.train(
        CLASSIFIER_PARAMS,
        final_data,
        num_boost_round=n_estimators,
    )
    return final_model, label_map, cv_dir_acc


def predict_direction_classifier(model, label_map, X):
    inv_map = {v: k for k, v in label_map.items()}
    preds = model.predict(X)
    class_idx = np.argmax(preds, axis=1)
    return np.array([inv_map[i] for i in class_idx])


def predict_direction_classifier_proba(model, label_map, X, label_order=DIRECTION_LABELS):
    """Returns an (n_samples, 3) probability matrix with columns reordered to
    match label_order, regardless of the label_map's internal index order."""
    probs = model.predict(X)
    inv_map = {v: k for k, v in label_map.items()}
    col_order = [inv_map[i] for i in range(len(inv_map))]  # label at each raw column index
    reordered = np.zeros_like(probs)
    for target_col_i, label in enumerate(label_order):
        source_col_i = col_order.index(label)
        reordered[:, target_col_i] = probs[:, source_col_i]
    return reordered


if __name__ == "__main__":
    feature_table_path = PROCESSED_DIR / "feature_table.csv"
    df = pd.read_csv(feature_table_path, parse_dates=["week_start"])

    train_df, val_df, test_df = chronological_split(df)
    train_val_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"Train+Val (used for CV): {len(train_val_df)}  Test (held out): {len(test_df)}")
    test_thresholds = get_category_thresholds(test_df)
    if USE_CATEGORY_THRESHOLDS:
        print("Per-category stable thresholds (active):")
        for category, threshold in sorted(CATEGORY_STABLE_THRESHOLDS.items()):
            print(f"  {category}: +-{threshold * 100:.2f}%")
    else:
        print(f"Pooled stable threshold (active): +-{STABLE_THRESHOLD * 100:.2f}% "
              f"(per-category thresholds tested and reverted — see config.py)")
    print()

    naive_preds = naive_persistence_forecast(test_df)
    evaluate("Naive Persistence", test_df["target"].values, naive_preds, threshold=test_thresholds)
    print()

    lgb_model, lgb_cv_acc = train_lightgbm_cv(train_val_df)
    lgb_preds = lgb_model.predict(test_df[FEATURE_COLS])
    evaluate("LightGBM (test)", test_df["target"].values, lgb_preds, threshold=test_thresholds)
    print("\nFeature importances (LightGBM 'gain'-based split usage):")
    importances = pd.Series(lgb_model.feature_importance(importance_type="gain"), index=FEATURE_COLS).sort_values(ascending=False)
    print(importances.astype(int))
    print()

    ridge_model, ridge_cv_acc = train_ridge_cv(train_val_df)
    ridge_preds = ridge_model.predict(test_df[FEATURE_COLS])
    evaluate("Ridge (test)", test_df["target"].values, ridge_preds, threshold=test_thresholds)
    print()

    clf_model, label_map, clf_cv_acc = train_direction_classifier_cv(train_val_df)
    clf_preds_dir = predict_direction_classifier(clf_model, label_map, test_df[FEATURE_COLS])
    true_dir = direction_from_pct_change(test_df["target"].values, threshold=test_thresholds)
    evaluate_classifier("Direction Classifier (test)", true_dir, clf_preds_dir)
    print("\nFeature importances (Direction Classifier):")
    clf_importances = pd.Series(clf_model.feature_importance(importance_type="gain"), index=FEATURE_COLS).sort_values(ascending=False)
    print(clf_importances.astype(int))
    print()

    lgb_dir = direction_from_pct_change(lgb_preds, threshold=test_thresholds)
    ridge_dir = direction_from_pct_change(ridge_preds, threshold=test_thresholds)

    ensemble_dir = ensemble_direction(lgb_dir, ridge_dir, clf_preds_dir)
    evaluate_classifier("Ensemble (test)", true_dir, ensemble_dir)

    results = pd.DataFrame([
        {"model": "Naive", "directional_accuracy": directional_accuracy(test_df["target"].values, naive_preds, threshold=test_thresholds)},
        {"model": "LightGBM", "directional_accuracy": directional_accuracy(test_df["target"].values, lgb_preds, threshold=test_thresholds)},
        {"model": "Ridge", "directional_accuracy": directional_accuracy(test_df["target"].values, ridge_preds, threshold=test_thresholds)},
        {"model": "Direction Classifier", "directional_accuracy": (true_dir == clf_preds_dir).mean()},
        {"model": "Ensemble", "directional_accuracy": (true_dir == ensemble_dir).mean()},
    ])
    out_path = PROCESSED_DIR / "model_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")