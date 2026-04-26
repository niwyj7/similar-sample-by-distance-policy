
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# ==========================================
# 0. Environment & Database Initialization
# ==========================================
# sys.path.append("/")
# import DataLoader

# ed = DataLoader('location')
# esql = Energy("location", "location_Weather")

# ==========================================
# 1. Data Acquisition Module
# ==========================================
def get_orderbook_data(start_time, end_time):
    base_dir = "/data1/..."
    dfs = [pd.read_parquet(os.path.join(root, fn)) 
           for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d, "T2"))
           for root, _, files in os.walk(os.path.join(base_dir, d, "T2")) 
           for fn in files if fn.endswith(".parquet")]
    
    if not dfs: return pd.DataFrame()
    trade_his = pd.concat(dfs, ignore_index=False, sort=False)

    def _extract_level(x, is_first=True):
        if isinstance(x, str): 
            try:
                x = eval(x)
            except Exception:
                return np.nan, np.nan
                
        if isinstance(x, (list, tuple)) and len(x) > 0:
            if is_first:
                return float(x[0][0]), float(x[0][1])
            else:
                return float(x[-1][0]), float(x[-1][1])
        return np.nan, np.nan

    # Extract order book prices and calculate the mid-price. 
    # Using .tolist() avoids pandas ValueError caused by index misalignment during assignment.
    trade_his[["bid_price", "bid_amount"]] = trade_his["bids"].apply(lambda x: _extract_level(x, True)).tolist()
    trade_his[["ask_price", "ask_amount"]] = trade_his["asks"].apply(lambda x: _extract_level(x, False)).tolist()
    
    trade_his["bid_price"] = (trade_his["bid_price"] + trade_his["ask_price"]) / 2

    return trade_his.loc[start_time:end_time]

def prepare_train_data(start_date, end_date):
    trade_his = get_orderbook_data(start_date, end_date)
    rt_real = ed.pull(['da', 'rt'], start=start_date, end=end_date).rename_axis('time')
    
    # Align order book and real-time prices to calculate the price spread.
    # Spread = bid_price - real_time_price. The sign indicates arbitrage direction.
    common_idx = trade_his.index.intersection(rt_real.index)
    aligned = trade_his.loc[common_idx, ["bid_price"]].join(rt_real.loc[common_idx, ["rt"]], how="inner")
    
    aligned['hour'] = aligned.index.hour
    aligned['prc_diff'] = aligned['bid_price'] - aligned['rt']
    aligned['prc_diff_sign'] = np.sign(aligned['prc_diff'])    
    return aligned.dropna()

# ==========================================
# 2. Feature Engineering: Mutual Information (MI) Weights
# ==========================================
def var_importance_mi_by_hour(hist_df, features):
    # Filter out samples with missing values or zero price spread to ensure valid classification targets.
    df0 = hist_df.dropna(subset=['prc_diff_sign'] + features).copy()
    df0 = df0[df0['prc_diff_sign'] != 0]
    rows = []
    
    for h in range(24):
        d = df0[df0["hour"] == h]
        if d.empty: continue

        X = d[features].astype(float)
        # Standardize features (Z-score normalization). 
        # Math Reason: MI relies on k-Nearest Neighbors (KNN) for density estimation. 
        # Features must be scale-invariant so variables with larger absolute magnitudes do not dominate the distance metric.
        Xz = ((X - X.mean()) / X.std(ddof=0).replace(0, np.nan)).fillna(0)
        y_h = d['prc_diff_sign'].astype(int)

        # Minimum sample check for statistical validity.
        if (y_h == 1).sum() < 2 or (y_h == -1).sum() < 2: 
            continue

        # Calculate Mutual Information scores. 
        # Math Reason: discrete_features=False treats weather data as continuous variables. 
        # Unlike ANOVA, MI captures non-linear dependencies (e.g., U-shaped relationships) between weather and price spreads.
        mi_scores = mutual_info_classif(Xz, y_h, discrete_features=False, random_state=42)
        
        for f, score in zip(features, mi_scores):
            rows.append({"hour": h, "feature": f, "weight": score})

    res_df = pd.DataFrame(rows)
    
    # Apply a minimal baseline weight (1e-4) to prevent division-by-zero errors 
    # during normalization if all MI scores in a given hour happen to be zero.
    if not res_df.empty:
        res_df['weight'] = res_df['weight'].clip(lower=1e-4)
        
    return res_df

# ==========================================
# 3. Core Strategy Logic
# ==========================================
def strategy_NN3(lookback: int, date: str, tn: int, z_threshold: float = 0.8, min_samples: int = 5):
    
    features = ['win100_spd', 't2', 'rhu', 'd2', 'sf', 'sp', 'tp', 'ssrd', 'u100', 'v100', 'tcc']
    end_date = pd.to_datetime(date)
    target_date = end_date + pd.Timedelta(days=int(tn))
    start_date = end_date - pd.Timedelta(days=int(lookback))

    # Fetch weather data and apply Z-score standardization.
    # We use historical mean and standard deviation to transform the target data 
    # to maintain distribution alignment (avoiding data leakage).
    w_hist = esql.select(features, start=start_date.strftime("%Y%m%d"), end=end_date.strftime("%Y%m%d"), NN=1).groupby("datetime").mean().dropna()
    w_tgt = esql.select(features, start=target_date.strftime("%Y%m%d"), end=target_date.strftime("%Y%m%d"), NN=tn).groupby("datetime").mean().dropna()
    
    w_mu, w_sigma = w_hist.mean(), w_hist.std().replace(0, np.nan)
    w_hist = ((w_hist - w_mu) / w_sigma).fillna(0)
    w_tgt = ((w_tgt - w_mu) / w_sigma).fillna(0)

    # Fetch historical price and order book data, then merge.
    prc_hist = prepare_train_data(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
    hist = prc_hist.join(w_hist, how="inner")
    if hist.empty: return pd.DataFrame()

    # Get dynamic feature weights based on Mutual Information.
    var_importance = var_importance_mi_by_hour(hist, features)
    rows = []

    for ts, xz in w_tgt.iterrows():
        h = ts.hour
        hist_hour = hist[hist.index.hour == h]
        
        # Extract and normalize weights so they sum to 1.
        vh = var_importance[var_importance["hour"] == h]
        _w = vh.set_index("feature")["weight"].reindex(features).fillna(1e-4)
        _w = _w / _w.sum() 
        
        # Calculate Weighted Euclidean Distance.
        # Math Reason: We multiply the difference array by sqrt(weights) before applying np.linalg.norm.
        # Because the L2 norm squares the components, squaring sqrt(w) results in sum(w_i * (x_i - y_i)^2).
        X_hist_arr = hist_hour[features].values
        v_arr = xz[features].values.reshape(1, -1)
        w_arr = np.sqrt(_w.values).reshape(1, -1) 
        dists = np.linalg.norm((X_hist_arr - v_arr) * w_arr, axis=1)

        # Core logic: Use absolute Z-score distance threshold.
        # Math Reason: dist <= z_threshold (e.g., 1.5) strictly guarantees that the selected historical 
        # matches have an average feature deviation of less than 1.5 standard deviations from the target weather.
        sim_indices = hist_hour.index[dists <= z_threshold]
        sim = hist.loc[sim_indices]
        
        # Extreme Weather Safeguard: 
        # If the number of matching days is mathematically insignificant (< min_samples), 
        # we refuse to trade (output 0) rather than forcing predictions on noisy/irrelevant data.
        if len(sim) < min_samples:
            rows.append({
                "ts": ts, 
                "n_sample": len(sim), 
                "buy_prob_neg": 0.0, 
                "sell_prob_pos": 0.0, 
                "Q_n2": 0.0, 
                "hour": h
            })
            continue
        sim = sim.sort_index()
        
        # Apply Time-Decay Weighting to the similar samples.
        # Math Reason: np.exp(np.linspace(-3, 0, n_sim)) generates an exponentially increasing weight curve.
        # More recent historical samples are given higher weights, acknowledging market regime shifts 
        # and baseline expectation changes over time.
        n_sim = len(sim)
        weights = np.exp(np.linspace(-3, 0, n_sim))
        weights /= weights.sum()
        
        # Calculate weighted probabilities for negative and positive spreads.
        prob_neg = float(np.sum((sim['prc_diff'] < 0) * weights))
        prob_pos = float(np.sum((sim['prc_diff'] > 0) * weights))
        
        # Asymmetric Signal Mapping for Imbalanced Classes.
        # Math Reason: Negative spreads are rare in electricity markets. Therefore, a 50% probability 
        # of a negative spread in similar historical regimes is highly significant. We lower the buy 
        # threshold for rare events, but demand high confidence (>70%) to short standard positive spreads.
        q = 0.0
        if prob_neg >= 0.5:  
            q = 1.0
        elif prob_pos > 0.7: 
            q = -1.0

        rows.append({
            "ts": ts, 
            "n_sample": len(sim), 
            "buy_prob_neg": prob_neg, 
            "sell_prob_pos": prob_pos, 
            "Q_n2": q, 
            "hour": h
        })

    return pd.DataFrame(rows).set_index("ts")

# ==========================================
# 4. Main Function & Execution Entry
# ==========================================
def main(pred_date: str):
    # lookback=40: Use past 40 days for historical context.
    # z_threshold=1.5: Tolerate up to a 1.5 standard deviation divergence in weighted weather similarity.
    # min_samples=3: Require at least 3 matching historical days to generate a signal.
    return strategy_NN3(lookback=40, date=pred_date, tn=2, z_threshold=1.5, min_samples=3)

if __name__ == "__main__":
    final_signal = pd.DataFrame()  
    
    # Define target prediction dates
    date_range = pd.date_range(start="2026-04-17", end="2026-04-17", freq='D')

    for single_date in date_range:
        pred = main(single_date.strftime("%Y%m%d"))
        if not pred.empty:
            print(f"\nPredictions for {single_date.strftime('%Y-%m-%d')}:")
            final_signal = pd.concat([final_signal, pred])

    if not final_signal.empty:
        print("\n=== Final aggregated predictions ===")
        print(final_signal)
        
        print("\n=== Strong BUY Signals (Negative Spread Expected) ===")
        # Filter and display only high-conviction long/negative spread buying opportunities
        buy_signals = final_signal[final_signal['Q_n2'] == 1.0]
        if not buy_signals.empty:
            print(buy_signals)
        else:
            print("No strong buy signals found for this period.")
