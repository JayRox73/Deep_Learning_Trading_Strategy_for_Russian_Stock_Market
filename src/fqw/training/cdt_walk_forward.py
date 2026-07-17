"""CDT walk-forward training with inline trading simulation."""

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks

from fqw.config import CDTConfig
from fqw.datasets.tensors_cdt import create_tensors_cdt
from fqw.metrics.classification import weighted_f_score_dict
from fqw.models.cdt import build_cdt_1d_cnn


def run_cdt_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    config: CDTConfig | None = None,
    confidence_threshold: float | None = None,
    alpha: float | None = None,
) -> dict | None:
    cfg = config or CDTConfig()
    confidence_threshold = (
        cfg.confidence_threshold if confidence_threshold is None else confidence_threshold
    )
    alpha = cfg.alpha if alpha is None else alpha

    print(f"\n{'=' * 70}\n  {model_name} ({len(feature_cols)} features)\n{'=' * 70}")
    x_all, y_all, dates_all, prices_all = create_tensors_cdt(
        df, feature_cols, cfg.window_size, alpha
    )
    if len(x_all) == 0:
        return None

    dt = pd.to_datetime(dates_all)
    train_end = dt.min() + relativedelta(months=cfg.train_months)
    ts = int((dt < train_end).sum())

    model = build_cdt_1d_cnn(
        n_features=x_all.shape[2],
        window_size=cfg.window_size,
        learning_rate=cfg.learning_rate,
        dropout=cfg.dropout,
        l2_decay=cfg.l2_decay,
    )

    y_tr = y_all[:ts]
    present = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=present, y=y_tr)
    cw_dict = dict(zip(present, cw))
    for c in (0, 1, 2):
        cw_dict.setdefault(c, 1.0)

    model.fit(
        x_all[:ts],
        y_tr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=0.05,
        class_weight=cw_dict,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ],
        verbose=0,
    )

    preds_all, acts_all, dates_out, prices_out = [], [], [], []
    cur = train_end
    while True:
        nxt = cur + relativedelta(months=cfg.test_months)
        mask = (dt >= cur) & (dt < nxt)
        xt, yt = x_all[mask], y_all[mask]
        if len(xt) == 0:
            break
        probs = model.predict(xt, verbose=0)
        pr = [
            1 if p[1] > confidence_threshold else 2 if p[2] > confidence_threshold else 0
            for p in probs
        ]
        preds_all.extend(pr)
        acts_all.extend(yt)
        dates_out.extend(dates_all[mask])
        prices_out.extend(prices_all[mask])
        model.fit(xt, yt, epochs=1, batch_size=cfg.batch_size, class_weight=cw_dict, verbose=0)
        cur = nxt

    yp, yt2, pa, da = map(np.array, (preds_all, acts_all, prices_out, dates_out))
    wfs = weighted_f_score_dict(yt2, yp, cfg.beta1, cfg.beta2, cfg.beta3)

    trades, eq = [], [100_000.0]
    in_pos, ep, pd2, ib = False, 0.0, 0, False
    for i in range(len(yp)):
        s, p = yp[i], pa[i]
        if in_pos:
            u = (p - ep) / ep * 100 if pd2 == 1 else (ep - p) / ep * 100
            if u >= cfg.be_trigger_pct and not ib:
                ib = True
            close = (pd2 == 1 and s == 2) or (pd2 == -1 and s == 1) or (ib and u <= 0)
            if close:
                pnl = ((p - ep) / ep if pd2 == 1 else (ep - p) / ep) - cfg.commission * 2
                eq.append(eq[-1] * (1 + pnl))
                trades.append(pnl * 100)
                in_pos, ib = False, False
                if s == 1:
                    in_pos, ep, pd2, ib = True, p, 1, False
                elif s == 2:
                    in_pos, ep, pd2, ib = True, p, -1, False
            else:
                eq.append(eq[-1])
        else:
            if s == 1:
                in_pos, ep, pd2, ib = True, p, 1, False
            elif s == 2:
                in_pos, ep, pd2, ib = True, p, -1, False
            eq.append(eq[-1])

    eq = np.array(eq)
    total_return = (eq[-1] / eq[0] - 1) * 100
    td = (pd.to_datetime(da[-1]) - pd.to_datetime(da[0])).days
    ty = max(td / 365.25, 0.01)
    dr = np.diff(eq) / eq[:-1]
    dr = dr[dr != 0]
    sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252 * 78) if len(dr) > 0 and np.std(dr) > 0 else 0
    nt = len(trades)
    if nt:
        pnls = np.array(trades)
        wr = np.sum(pnls > 0) / nt * 100
        gp, gl = np.sum(pnls[pnls > 0]), abs(np.sum(pnls[pnls < 0]))
        pf = gp / gl if gl > 0 else float("inf")
        aw = np.mean(pnls[pnls > 0]) if np.any(pnls > 0) else 0
        al = abs(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0
        rr = aw / al if al > 0 else float("inf")
    else:
        wr = pf = rr = 0

    return {
        "model_name": model_name,
        "total_return": total_return,
        "annual_return": ((eq[-1] / eq[0]) ** (1 / ty) - 1) * 100,
        "sharpe": sharpe,
        "wfs": wfs["WFS"],
        "accuracy": wfs["Accuracy"],
        "profit_factor": pf,
        "rr_ratio": rr,
        "win_rate": wr,
        "n_trades": nt,
        "equity_curve": eq,
    }
