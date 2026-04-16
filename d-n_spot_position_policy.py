import pandas as pd
import numpy as np
import os
import sys
from functools import lru_cache
# import your dataloader
# ed = EnergyDataLoader('lcation')
# esql = EnergySQL("location", "location_Weather")

WEATHER_FEATURES = [
    'win100_spd', 't2', 'rhu', 'd2', 'sp', 'tp', 'ssrd', 'tcc', 'u100', 'v100', 'sf'
]

def get_weather_data(start_time: str, end_time: str, NN: int = 1) -> pd.DataFrame:
    return (
        # your weather features
        # 
    )


def get_orderbook_data(start_time: str, end_time: str) -> pd.DataFrame:
    """
    oderbook data return deal price
    """

    return trade_his.loc[start_time:end_time]


def get_rt_data(start_time: str, end_time: str) -> pd.DataFrame:

    price_data = ed.pull(['rt'], start=start_time, end=end_time)
    price_data.index = pd.to_datetime(price_data.index)
    return price_data.loc[start_time:end_time]


def prepare_train_data(start_date: str, end_date: str) -> pd.DataFrame:
 
    trade_his = get_orderbook_data(start_date, end_date)
    rt_real = get_rt_data(start_date, end_date)

    trade_cols = ["bid_price", "bid_amount", "ask_price", "ask_amount"]
    common_idx = trade_his.index.intersection(rt_real.index)

    aligned = (
        trade_his.loc[common_idx, trade_cols]
        .join(rt_real.loc[common_idx, ["rt"]].rename(columns={"rt": "real_rt"}), how="inner")
        .sort_index()
    )
    aligned["hour"] = aligned.index.hour
    aligned["prc_diff_t2"] = aligned["bid_price"] - aligned["real_rt"]
    aligned["prc_diff_sign_t2"] = np.sign(aligned["prc_diff_t2"])
    return aligned

def _anova_f_2class(x: pd.Series, y: pd.Series):
    """
    one-way ANOVA F stats, eta² 
   
    """
    x0, x1 = x[y == -1].dropna(), x[y == 1].dropna()
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return np.nan, np.nan

    m0, m1 = x0.mean(), x1.mean()
    m = (n0 * m0 + n1 * m1) / (n0 + n1)
    ss_between = n0 * (m0 - m) ** 2 + n1 * (m1 - m) ** 2
    ss_within = (n0 - 1) * x0.var(ddof=1) + (n1 - 1) * x1.var(ddof=1)

    if ss_within <= 0 or (n0 + n1 - 2) <= 0:
        return np.nan, np.nan

    F = (ss_between / 1) / (ss_within / (n0 + n1 - 2))
    eta2 = ss_between / (ss_between + ss_within)
    return float(F), float(eta2)


def var_importance_by_hour(df: pd.DataFrame, features: list) -> pd.DataFrame:
  
    df = df[features + ["prc_diff_sign_t2"]].copy()
    df = df.dropna(subset=["prc_diff_sign_t2"])
    df = df[df["prc_diff_sign_t2"] != 0].copy()
    df["prc_diff_sign_t2"] = df["prc_diff_sign_t2"].astype(int)
    df["hour"] = df.index.hour

    rows = []
    for h in range(24):
        d = df[df["hour"] == h].dropna(subset=features).copy()
        if d.empty:
            continue

        # hourly z-score standardise
        X = d[features].astype(float)
        Xz = ((X - X.mean()) / X.std(ddof=0).replace(0, np.nan)).dropna()
        y_h = d.loc[Xz.index, "prc_diff_sign_t2"]

        if (y_h == 1).sum() < 2 or (y_h == -1).sum() < 2:
            continue

        for f in features:
            F, eta2 = _anova_f_2class(Xz[f], y_h)
            rows.append({
                "hour": h, "feature": f, "F": F, "eta2": eta2,
                "n_pos": int((y_h == 1).sum()),
                "n_neg": int((y_h == -1).sum()),
            })

    return (
        pd.DataFrame(rows)
        .dropna(subset=["F", "eta2"])
        .rename(columns={"eta2": "variance_explained(eta2)"})
    )


def _get_hour_weights(var_importance: pd.DataFrame, features: list, hour: int) -> np.ndarray:

    vh = var_importance[var_importance["hour"] == hour]
    if vh.empty:
        return np.ones(len(features)) / len(features)

    w = (
        vh.set_index("feature")["variance_explained(eta2)"]
        .reindex(features)
        .fillna(0.0)
    )
    total = w.sum()
    return (w / total).values if total > 0 else np.ones(len(features)) / len(features)


def _weighted_euclidean(X: np.ndarray, v: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    加权欧氏距离：sqrt( sum( w_i * (x_i - v_i)^2 ) )
    X: (n_samples, n_features), v: (1, n_features), weights: (n_features,)
    """
    diff = X - v  # (n_samples, n_features)
    return np.sqrt((diff ** 2 * weights).sum(axis=1))


def _map_prob_to_position(buy_prob: float, sell_prob: float,
                          n_sample: int, min_samples: int,
                          n_pos: int, n_neg: int) -> float:
    """
    Continuous Position Mapping:
      - Base signal: buy_prob - sell_prob, range [-1, 1]
      - Sample size confidence scaling: Position size scales down as the sample size decreases.
      - Class imbalance penalty: Reduces confidence when the ratio of positive to negative samples is severely skewed.
    Returns the position coefficient: positive = buy, negative = sell, 0 = no action.
    """
    # siample number confidence adjustment
    sample_conf = min(1.0, n_sample / max(min_samples * 2, 20))

    # class balance confidence（ps and neg ratio > 4:1 penalty）
    total = n_pos + n_neg
    if total > 0:
        balance_ratio = min(n_pos, n_neg) / total
        balance_conf = min(1.0, balance_ratio / 0.2)  
    else:
        balance_conf = 0.0

    q_raw = buy_prob - sell_prob  # [-1, 1]
    return round(q_raw * sample_conf * balance_conf, 3)


def strategy_N(
    lookback: int,
    date: str,
    tn: int,
    features: list = None,
    min_samples: int = 5,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Strategy for predicting the direction of real-time electricity prices based on similar weather days.

    Parameters
    ----------
    lookback  : Number of historical lookback days
    date      : Backtesting base date (format YYYYMMDD)
    tn        : Number of days ahead for prediction (1=tomorrow, 2=day after tomorrow)
    features  : List of meteorological features used
    min_samples: Minimum number of similar samples per hour
    threshold : Similarity threshold coefficient (larger value means a wider neighbourhood)

    Returns
    -------
    DataFrame, with target day timestamps as the index, containing the following columns:
      hour, n_sample, buy_prob, sell_prob, position, n_pos, n_neg, balance_ratio
    """
    if features is None:
        features = WEATHER_FEATURES

    end_date = pd.to_datetime(date)
    target_date = end_date + pd.Timedelta(days=int(tn))
    start_date = end_date - pd.Timedelta(days=int(lookback))

    # ── 1. get weather ──
    w_hist = get_weather_data(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), NN=1)
    w_tgt = get_weather_data(target_date.strftime("%Y%m%d"), target_date.strftime("%Y%m%d"), NN=tn)

    w_hist = w_hist.dropna(subset=features)
    w_tgt = w_tgt.dropna(subset=features)

    # ── 2. standardise──
    w_mu = w_hist[features].mean()
    w_sigma = w_hist[features].std().replace(0, np.nan)
    w_hist[features] = ((w_hist[features] - w_mu) / w_sigma).fillna(0)
    w_tgt[features] = ((w_tgt[features] - w_mu) / w_sigma).fillna(0)

    # ── 3. historical price──
    prc_hist = prepare_train_data(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))

    # ── 4. merge weather and price──
    hist = prc_hist.join(w_hist, how="inner")
    if hist.empty:
        print("[WARN] content ")
        return pd.DataFrame()

    # ── 5. ANOVA ──
    var_importance = var_importance_by_hour(hist, features)

    # ── 6. predict  ──
    rows = []
    for ts, xz in w_tgt.iterrows():
        h = ts.hour
        hist_hour_idx = hist.index[hist.index.hour == h]

        if len(hist_hour_idx) < min_samples:
            rows.append({
                "ts": ts, "hour": h, "n_sample": 0,
                "buy_prob": 0.0, "sell_prob": 0.0, "position": 0.0,
                "n_pos": 0, "n_neg": 0, "balance_ratio": 0.0,
            })
            continue

        # ── 6a.  ANOVA hourly──
        weights = _get_hour_weights(var_importance, features, h)

        # ── 6b. Eucleadian distance ──
        X_hist_z = w_hist.loc[hist_hour_idx].values
        v = xz[features].values.reshape(1, -1)
        dists = _weighted_euclidean(X_hist_z, v, weights)

        # ── 6c. similar sample ──
        thr = np.min(dists) + threshold * np.std(dists)
        sim_mask = dists <= thr
        sim_indices = hist_hour_idx[sim_mask]
        sim = hist.loc[sim_indices]
        n = len(sim)

        if n > 3:
            # print similar sample
            sim_dist = pd.Series(dists[sim_mask], index=sim_indices).sort_values()
            top5_days = sim_dist.index[:5].normalize().strftime("%Y-%m-%d").unique()
            print(f"[{ts:%Y-%m-%d %H}:00] top-5 similar days: {', '.join(top5_days)}")

            n_pos = int((sim["prc_diff_sign_t2"] == 1).sum())
            n_neg = int((sim["prc_diff_sign_t2"] == -1).sum())
            buy_prob = float((sim["prc_diff_t2"] < 0).mean())   # bid < rt → buy
            sell_prob = float((sim["prc_diff_t2"] > 0).mean())  # bid > rt → sell
            balance_ratio = min(n_pos, n_neg) / max(n_pos + n_neg, 1)
        else:
            buy_prob = sell_prob = 0.0
            n_pos = n_neg = 0
            balance_ratio = 0.0

        position = _map_prob_to_position(buy_prob, sell_prob, n, min_samples, n_pos, n_neg)

        rows.append({
            "ts": ts, "hour": h, "n_sample": n,
            "buy_prob": round(buy_prob, 4),
            "sell_prob": round(sell_prob, 4),
            "position": position,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "balance_ratio": round(balance_ratio, 4),
        })

    return pd.DataFrame(rows).set_index("ts")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

def main(pred_date: str) -> pd.DataFrame:
    """
    pred_date : YYYYMMDD
    """
    return strategy_N(lookback=30, date=pred_date, tn=2, threshold=5.0)


if __name__ == "__main__":
    date_range = pd.date_range(start="2026-04-01", end="2026-04-16", freq="D")

    results = []
    for single_date in date_range:
        date_str = single_date.strftime("%Y%m%d")
        print(f"\n{'='*40}")
        print(f"Running prediction for base date: {date_str}")
        pred = main(date_str)
        if not pred.empty:
            results.append(pred)
        print(pred.to_string())

    if results:
        final_signal = pd.concat(results)
        print("\n" + "="*40)
        print("Final aggregated predictions:")
        print(final_signal.to_string())
