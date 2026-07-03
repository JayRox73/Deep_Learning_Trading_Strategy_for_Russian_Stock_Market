"""Record-based walk-forward training for MLP."""

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from fqw.config import MLPConfig
from fqw.datasets.tensors_mlp import create_tensors_mlp
from fqw.models.mlp import build_mlp_model


def run_moving_window_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    config: MLPConfig | None = None,
    alpha: float | None = None,
) -> pd.DataFrame | None:
    """Record-count walk-forward that outputs probabilities for threshold sweep."""
    cfg = config or MLPConfig()
    alpha = cfg.alpha if alpha is None else alpha

    print(f"\n{'=' * 70}\n  RUN: {model_name}\n{'=' * 70}")

    x_raw, y_raw, dates_raw, prices_raw = create_tensors_mlp(
        df, feature_cols=feature_cols, window_size=cfg.window_size, alpha=alpha
    )
    n = len(x_raw)
    need = cfg.train_records + cfg.val_records + cfg.test_records
    if n < need:
        print("⚠️ Insufficient data")
        return None

    all_actuals, all_prices, all_dates = [], [], []
    all_probs_up, all_probs_down = [], []

    start = 0
    while start + need <= n:
        train_end = start + cfg.train_records
        val_end = train_end + cfg.val_records
        test_end = val_end + cfg.test_records

        scaler = StandardScaler()
        n_feat = x_raw.shape[2]
        scaler.fit(x_raw[start:train_end].reshape(-1, n_feat))

        x_train = scaler.transform(x_raw[start:train_end].reshape(-1, n_feat)).reshape(-1, cfg.window_size, n_feat)
        x_val = scaler.transform(x_raw[train_end:val_end].reshape(-1, n_feat)).reshape(-1, cfg.window_size, n_feat)
        x_test = scaler.transform(x_raw[val_end:test_end].reshape(-1, n_feat)).reshape(-1, cfg.window_size, n_feat)
        y_train = y_raw[start:train_end]
        y_val = y_raw[train_end:val_end]

        model = build_mlp_model(
            input_shape=(cfg.window_size, n_feat),
            learning_rate=cfg.learning_rate,
            dropout=cfg.dropout,
            l2_decay=cfg.l2_decay,
        )
        classes = np.unique(y_train)
        class_weight_dict = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))

        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            class_weight=class_weight_dict,
            callbacks=[es],
            verbose=0,
        )

        probs = model.predict(x_test, batch_size=cfg.batch_size, verbose=0)
        all_probs_up.extend(probs[:, 1])
        all_probs_down.extend(probs[:, 2])
        all_actuals.extend(y_raw[val_end:test_end])
        all_prices.extend(prices_raw[val_end:test_end])
        all_dates.extend(dates_raw[val_end:test_end])

        start += cfg.step_records
        del model
        tf.keras.backend.clear_session()

    return pd.DataFrame(
        {
            "DateTime": all_dates,
            "Price": all_prices,
            "Actual": all_actuals,
            "Prob_Up": all_probs_up,
            "Prob_Down": all_probs_down,
        }
    )
