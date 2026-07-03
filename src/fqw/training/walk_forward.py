"""Walk-forward training and prediction."""

import os
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from fqw.config import CNNConfig
from fqw.datasets.tensors import create_tensors
from fqw.models.cnn import build_cnn_model


def run_backtest_with_fine_tuning(
    df: pd.DataFrame,
    ticker_name: str,
    alpha: float,
    confidence_threshold: float,
    config: CNNConfig | None = None,
    save_plots: bool = False,
    best_config: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Month-based walk-forward CNN training with monthly fine-tuning."""
    cfg = config or CNNConfig()

    if save_plots:
        os.makedirs(f"results/plots/{ticker_name}", exist_ok=True)
        os.makedirs(f"results/metrics/{ticker_name}", exist_ok=True)

    total_months = (df["DateTime"].max() - df["DateTime"].min()).days // 30
    if total_months < cfg.train_months + cfg.test_months:
        print(f"⚠️ {ticker_name}: insufficient data")
        return pd.DataFrame(), {}

    cols_to_exclude = {"DateTime", "Target", "index"}
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]

    x_all, y_all, dates_all, prices_all, _ = create_tensors(
        df, window_size=cfg.window_size, alpha=alpha, feature_cols=feature_cols
    )
    if len(x_all) == 0:
        return pd.DataFrame(), {}

    dt_index = pd.to_datetime(dates_all)
    start_date = dt_index.min()
    current_test_start = start_date + DateOffset(months=cfg.train_months)

    model = build_cnn_model(
        input_shape=(cfg.window_size, x_all.shape[2]),
        learning_rate=cfg.learning_rate,
        dropout=cfg.dropout,
    )

    initial_mask = (dt_index >= start_date) & (dt_index < current_test_start)
    if initial_mask.sum() < 10:
        return pd.DataFrame(), {}

    x_train_init, y_train_init = x_all[initial_mask], y_all[initial_mask]
    classes = np.unique(y_train_init)
    class_weight_dict = dict(
        zip(classes, compute_class_weight("balanced", classes=classes, y=y_train_init))
    )

    print(
        f"Training {ticker_name} on {initial_mask.sum()} samples. "
        f"Class weights: {class_weight_dict}"
    )
    model.fit(
        x_train_init,
        y_train_init,
        epochs=cfg.initial_epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        class_weight=class_weight_dict,
    )

    all_preds: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []

    while current_test_start < dt_index.max():
        test_end = current_test_start + DateOffset(months=cfg.test_months)
        test_mask = (dt_index >= current_test_start) & (dt_index < test_end)

        x_test, y_test = x_all[test_mask], y_all[test_mask]
        if len(x_test) == 0:
            current_test_start = test_end
            continue

        probs = model.predict(x_test, verbose=0)
        final_classes = []
        confident_mask = []
        for p in probs:
            is_confident = (p[1] > confidence_threshold) or (p[2] > confidence_threshold)
            confident_mask.append(is_confident)
            if p[1] > confidence_threshold:
                final_classes.append(1)
            elif p[2] > confidence_threshold:
                final_classes.append(2)
            else:
                final_classes.append(0)

        confident_mask_arr = np.array(confident_mask)
        final_classes_arr = np.array(final_classes)

        chunk_metrics: dict[str, Any] = {
            "ticker": ticker_name,
            "alpha": alpha,
            "confidence_threshold": confidence_threshold,
            "period_start": current_test_start,
            "period_end": test_end,
            "total_samples": len(y_test),
            "confident_samples": int(confident_mask_arr.sum()),
            "confident_ratio": float(confident_mask_arr.mean()),
            "accuracy_all": float(np.mean(final_classes_arr == y_test)),
            "f1_macro_all": float(
                f1_score(y_test, final_classes_arr, average="macro", zero_division=0)
            ),
        }

        if confident_mask_arr.sum() > 10:
            y_test_conf = y_test[confident_mask_arr]
            y_pred_conf = final_classes_arr[confident_mask_arr]
            chunk_metrics.update(
                {
                    "accuracy_confident": float(np.mean(y_pred_conf == y_test_conf)),
                    "f1_macro_confident": float(
                        f1_score(y_test_conf, y_pred_conf, average="macro", zero_division=0)
                    ),
                    "precision_buy": float(
                        precision_score(
                            y_test_conf, y_pred_conf, labels=[1], average="micro", zero_division=0
                        )
                    ),
                    "recall_buy": float(
                        recall_score(
                            y_test_conf, y_pred_conf, labels=[1], average="micro", zero_division=0
                        )
                    ),
                    "f1_buy": float(
                        f1_score(
                            y_test_conf, y_pred_conf, labels=[1], average="micro", zero_division=0
                        )
                    ),
                    "precision_sell": float(
                        precision_score(
                            y_test_conf, y_pred_conf, labels=[2], average="micro", zero_division=0
                        )
                    ),
                    "recall_sell": float(
                        recall_score(
                            y_test_conf, y_pred_conf, labels=[2], average="micro", zero_division=0
                        )
                    ),
                    "f1_sell": float(
                        f1_score(
                            y_test_conf, y_pred_conf, labels=[2], average="micro", zero_division=0
                        )
                    ),
                }
            )
        else:
            for key in [
                "accuracy_confident",
                "f1_macro_confident",
                "precision_buy",
                "recall_buy",
                "f1_buy",
                "precision_sell",
                "recall_sell",
                "f1_sell",
            ]:
                chunk_metrics[key] = np.nan

        chunk_results = pd.DataFrame(
            {
                "DateTime": dates_all[test_mask],
                "Price": prices_all[test_mask],
                "Actual": y_test,
                "Predicted": final_classes_arr,
                "is_confident": confident_mask_arr,
                "Prob_Flat": probs[:, 0],
                "Prob_Buy": probs[:, 1],
                "Prob_Sell": probs[:, 2],
            }
        )
        for key, value in chunk_metrics.items():
            chunk_results[key] = value

        all_preds.append(chunk_results)
        all_metrics.append(chunk_metrics)

        model.fit(
            x_test,
            y_test,
            epochs=cfg.fine_tune_epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            class_weight=class_weight_dict,
        )
        current_test_start = test_end

    if not all_preds:
        return pd.DataFrame(), {}

    results_df = pd.concat(all_preds, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    summary_metrics: dict[str, Any] = {
        "ticker": ticker_name,
        "alpha": alpha,
        "confidence_threshold": confidence_threshold,
        "total_periods": len(metrics_df),
        "total_samples": int(metrics_df["total_samples"].sum()),
        "total_confident_samples": int(metrics_df["confident_samples"].sum()),
        "avg_confident_ratio": float(metrics_df["confident_ratio"].mean()),
    }

    total_conf = summary_metrics["total_confident_samples"]
    if total_conf > 0:
        valid_conf = metrics_df.dropna(subset=["accuracy_confident"])
        if not valid_conf.empty:
            summary_metrics["accuracy_confident_weighted"] = float(
                (valid_conf["accuracy_confident"] * valid_conf["confident_samples"]).sum()
                / valid_conf["confident_samples"].sum()
            )

    _ = best_config  # reserved for plot-saving hooks
    return results_df, summary_metrics
