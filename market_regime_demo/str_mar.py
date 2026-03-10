# =========================================================
# HYBRID MARKET CORE
# =========================================================

def _market_core(
    ticker="^NSEI",
    start_date="2010-01-01",
    long_horizon=30,
    mid_horizon=10
):
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.mixture import GaussianMixture
    import warnings
    warnings.filterwarnings("ignore")

    EPS = 1e-6
    REGIME_MAP = {0: "Expansion", 1: "Pullback", 2: "Correction", 3: "Crash"}

    df = yf.download(ticker, start=start_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']].dropna()
    df.index = pd.to_datetime(df.index)

    vix = yf.download("^INDIAVIX", start=start_date, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vix = vix[['Close']].rename(columns={"Close": "VIX"})
    vix.index = pd.to_datetime(vix.index)
    df = df.join(vix, how="inner")

    df["VIX_z"] = (df["VIX"] - df["VIX"].rolling(60).mean()) / (df["VIX"].rolling(60).std() + EPS)
    df["VIX_slope"] = df["VIX"] - df["VIX"].shift(5)
    df["VIX_accel"] = df["VIX"].pct_change(5) - df["VIX"].pct_change(10)

    df['ret_1d'] = df['Close'].pct_change()
    df['ret_5d'] = df['Close'].pct_change(5)

    df['vol_20'] = df['ret_1d'].rolling(20).std()
    df['vol_60'] = df['ret_1d'].rolling(60).std()

    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()

    df['Trend_Strength'] = (df['EMA_20'] - df['EMA_50']) / (df['EMA_50'] + EPS)
    df['Slope_20'] = (df['Close'] - df['Close'].shift(20)) / 20
    df['Slope_50'] = (df['Close'] - df['Close'].shift(50)) / 50

    df['Direction_Bias_Score'] = (
        np.sign(df['Trend_Strength']) +
        np.sign(df['Slope_20']) +
        np.sign(df['ret_5d'])
    )

    df['Slope_Decay'] = df['Slope_20'] - df['Slope_50']

    df['Structure_Stress'] = (
        df['Slope_Decay'].abs() +
        (df['Slope_20'] - df['Slope_20'].shift(10)).abs()
    )

    df['Structure_Stress'] = (
        df['Structure_Stress'] -
        df['Structure_Stress'].rolling(100).mean()
    ) / df['Structure_Stress'].rolling(100).std()

    df['Volatility_Stress'] = df['vol_20'] / (df['vol_60'] + EPS)

    df.dropna(inplace=True)

    stress_features = [
        'Structure_Stress','Slope_20','Slope_Decay',
        'Trend_Strength','Volatility_Stress',
        'VIX_z','VIX_slope','VIX_accel'
    ]

    price_features = stress_features + ['Direction_Bias_Score']

    # ===== HYBRID ADDITION =====
    gmm = GaussianMixture(n_components=4, random_state=42)
    df["unsupervised_regime"] = gmm.fit_predict(df[stress_features])

    def train_regime(h):
        future_min = df['Close'].rolling(h).min().shift(-h)
        drawdown = (future_min - df['Close']) / df['Close']

        y = np.zeros(len(df))
        y[(drawdown <= -0.03) & (drawdown > -0.05)] = 1
        y[(drawdown <= -0.05) & (drawdown > -0.08)] = 2
        y[drawdown <= -0.08] = 3

        model = LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=350,
            learning_rate=0.025,
            max_depth=4,
            random_state=42,
            verbose=-1
        )

        model.fit(
            df[stress_features + ['Direction_Bias_Score','unsupervised_regime']],
            y
        )

        return model

    regime_10d = train_regime(mid_horizon)
    regime_30d = train_regime(long_horizon)

    def train_price(h):
        tmp = df.copy()
        tmp['fwd_ret'] = tmp['Close'].pct_change(h).shift(-h)
        tmp.dropna(inplace=True)

        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.035,
            max_depth=4,
            random_state=42,
            verbose=-1
        )

        model.fit(
            tmp[price_features + ['unsupervised_regime']],
            tmp['fwd_ret']
        )

        return model

    price_10d = train_price(mid_horizon)
    price_30d = train_price(long_horizon)

    return (
        df,
        stress_features,
        regime_10d,
        regime_30d,
        price_10d,
        price_30d,
        REGIME_MAP
    )


# =========================================================
# MARKET REPORT (UNCHANGED)
# =========================================================

def market_report(
    ticker="^NSEI",
    start_date="2010-01-01",
    track_days=5,
    intraday_days=7
):
    import pandas as pd
    import yfinance as yf

    df, stress_features, regime_10d, regime_30d, price_10d, price_30d, REGIME_MAP = _market_core(
        ticker, start_date
    )

    rows, labels = [], []

    def snapshot(idx):
        x = df.iloc[idx]
        # X = x[stress_features + ['Direction_Bias_Score']].to_frame().T
        model_features = stress_features + ['Direction_Bias_Score', 'unsupervised_regime']
        X = x[model_features].to_frame().T


        return {
            "close": round(float(x['Close']), 2),
            "10D_expected_price": round(x['Close'] * (1 + price_10d.predict(X)[0]), 2),
            "30D_expected_price": round(x['Close'] * (1 + price_30d.predict(X)[0]), 2),
            "10D_Crash_Prob": round(regime_10d.predict_proba(X)[0][3], 3),
            "30D_Crash_Prob": round(regime_30d.predict_proba(X)[0][3], 3),
        }

    for i in range(track_days, 0, -1):
        rows.append(snapshot(-i))
        labels.append(f"T-{i}")

    rows.append(snapshot(-1))
    labels.append("Today")

    change_tracker = pd.DataFrame(rows, index=labels)
    change_tracker_delta = change_tracker.diff()

    df_5m = yf.download(
        ticker,
        interval="5m",
        period=f"{intraday_days}d",
        progress=False
    )

    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)

    df_5m = df_5m[['Close']].dropna()

    intraday_confirmation = (
        "Bullish"
        if df_5m['Close'].iloc[-1] > df_5m['Close'].iloc[-13]
        else "Bearish"
    )

    return {
        "change_tracker_5d": change_tracker,
        "change_tracker_delta": change_tracker_delta,
        "intraday_confirmation": intraday_confirmation,
    }


# =========================================================
# MARKET FEATURES (UNCHANGED)
# =========================================================

def market_features(
    ticker="^NSEI",
    start_date="2010-01-01"
):
    import pandas as pd

    df, stress_features, _, regime_30d, _, _, _ = _market_core(
        ticker=ticker,
        start_date=start_date
    )

    model_features = stress_features + ['Direction_Bias_Score', 'unsupervised_regime']


    rows = []
    for i in range(100, len(df)):
        x = df.iloc[i]
        X = x[model_features].to_frame().T
        probs = regime_30d.predict_proba(X)[0]

        rows.append({
            "date": df.index[i],
            "market_trend_strength": x["Trend_Strength"],
            "market_volatility_stress": x["Volatility_Stress"],
            "market_structure_stress": x["Structure_Stress"],
            "market_direction_score": x["Direction_Bias_Score"],
            "market_expansion_prob_30d": probs[0],
            "market_crash_prob_30d": probs[3],
        })

    return pd.DataFrame(rows).set_index("date")
# =========================================================
# MARKET STATE VECTOR (MODEL SELECTION LAYER)
# =========================================================

def market_state_vector(
    ticker="^NSEI",
    start_date="2010-01-01"
):
    import numpy as np

    df, stress_features, regime_10d, regime_30d, price_10d, price_30d, REGIME_MAP = _market_core(
        ticker=ticker,
        start_date=start_date
    )

    latest = df.iloc[-1]

    # Include unsupervised regime (HYBRID part)
    model_features = stress_features + ['Direction_Bias_Score','unsupervised_regime']
    X = latest[model_features].to_frame().T

    # -----------------------
    # Core Model Outputs
    # -----------------------

    crash_prob_30d = regime_30d.predict_proba(X)[0][3]
    crash_prob_10d = regime_10d.predict_proba(X)[0][3]

    expected_return_30d = price_30d.predict(X)[0]
    expected_return_10d = price_10d.predict(X)[0]

    unsup_regime = int(latest["unsupervised_regime"])

    # -----------------------
    # Percentile Context
    # -----------------------

    def pct_rank(series, value):
        return float((series < value).mean())

    trend_pct = pct_rank(df["Trend_Strength"], latest["Trend_Strength"])
    vol_pct = pct_rank(df["Volatility_Stress"], latest["Volatility_Stress"])
    struct_pct = pct_rank(df["Structure_Stress"], latest["Structure_Stress"])

    # -----------------------
    # Logical Interpretation
    # -----------------------

    if trend_pct > 0.65:
        trend_state = "Strong Trend"
    elif trend_pct < 0.35:
        trend_state = "Weak Trend"
    else:
        trend_state = "Neutral"

    if vol_pct > 0.7:
        volatility_state = "High Vol"
    elif vol_pct < 0.3:
        volatility_state = "Low Vol"
    else:
        volatility_state = "Normal Vol"

    if struct_pct > 0.7:
        structure_state = "Unstable"
    else:
        structure_state = "Stable"

    if crash_prob_30d > 0.5:
        risk_state = "High Risk"
    elif crash_prob_30d > 0.3:
        risk_state = "Moderate Risk"
    else:
        risk_state = "Low Risk"

    if expected_return_30d > 0.02:
        direction_state = "Bullish"
    elif expected_return_30d < -0.02:
        direction_state = "Bearish"
    else:
        direction_state = "Sideways"

    # -----------------------
    # FINAL STATE VECTOR
    # -----------------------

    return {
        # Raw Probabilities
        "crash_prob_10d": float(crash_prob_10d),
        "crash_prob_30d": float(crash_prob_30d),

        # Expected Returns
        "expected_return_10d": float(expected_return_10d),
        "expected_return_30d": float(expected_return_30d),

        # Percentile Context
        "trend_percentile": trend_pct,
        "volatility_percentile": vol_pct,
        "structure_percentile": struct_pct,

        # States
        "trend_state": trend_state,
        "volatility_state": volatility_state,
        "structure_state": structure_state,
        "risk_state": risk_state,
        "direction_state": direction_state,

        # Hybrid Layer
        "unsupervised_regime_cluster": unsup_regime
    }
# =========================================================
# ML 14-DAY AHEAD REGIME PREDICTION + PERFORMANCE
# =========================================================

def ml_next_regime_model(
    ticker="^NSEI",
    start_date="2010-01-01",
    horizon=14   # 👈 default 14 days ahead
):
    import pandas as pd
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, log_loss

    df, stress_features, _, _, _, _, _ = _market_core(
        ticker=ticker,
        start_date=start_date
    )

    # --------------------------------------------------
    # 1️⃣ Predict regime 14 days ahead
    # --------------------------------------------------
    df["future_regime"] = df["unsupervised_regime"].shift(-horizon)
    df.dropna(inplace=True)

    features = stress_features + [
        "Direction_Bias_Score",
        "unsupervised_regime"
    ]

    X = df[features]
    y = df["future_regime"]

    # --------------------------------------------------
    # 2️⃣ Time-based split
    # --------------------------------------------------
    split_idx = int(len(df) * 0.7)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)

    return {
        "prediction_horizon_days": horizon,
        "accuracy": float(acc),
        "log_loss": float(ll)
    }
# =========================================================
# FULL REGIME INTERPRETATION (WITH DIRECTION)
# =========================================================

def regime_interpretation(
    ticker="^NSEI",
    start_date="2010-01-01",
    forward_horizon=10
):
    import pandas as pd
    import numpy as np

    df, stress_features, regime_10d, regime_30d, price_10d, price_30d, REGIME_MAP = _market_core(
        ticker=ticker,
        start_date=start_date
    )

    regimes = df["unsupervised_regime"]
    current_regime = regimes.iloc[-1]

    # ------------------------------------------------
    # 1️⃣ Forward returns for direction context
    # ------------------------------------------------
    df["fwd_return"] = df["Close"].pct_change(forward_horizon).shift(-forward_horizon)

    # ------------------------------------------------
    # 2️⃣ Regime statistics
    # ------------------------------------------------
    regime_stats = (
        df.groupby("unsupervised_regime")[[
            "Trend_Strength",
            "Direction_Bias_Score",
            "Volatility_Stress",
            "Structure_Stress",
            "Volume",
            "fwd_return"
        ]]
        .mean()
    )

    current_stats = regime_stats.loc[current_regime]

    # ------------------------------------------------
    # 3️⃣ Transition Probability (Pure Frequency)
    # ------------------------------------------------
    df["next_regime"] = regimes.shift(-1)

    transition_counts = (
        df.groupby(["unsupervised_regime", "next_regime"])
          .size()
          .unstack(fill_value=0)
    )

    transition_probs = transition_counts.div(
        transition_counts.sum(axis=1), axis=0
    )

    next_probs = transition_probs.loc[current_regime]
    next_probs_no_self = next_probs.drop(current_regime)

    most_likely_next = next_probs_no_self.idxmax()
    next_stats = regime_stats.loc[most_likely_next]

    # ------------------------------------------------
    # 4️⃣ Feature Change
    # ------------------------------------------------
    change = next_stats - current_stats

    return {
        "current_regime_cluster": int(current_regime),

        "current_regime_characteristics": {
            "avg_trend_strength": float(current_stats["Trend_Strength"]),
            "avg_direction_bias": float(current_stats["Direction_Bias_Score"]),
            "avg_volatility_stress": float(current_stats["Volatility_Stress"]),
            "avg_structure_stress": float(current_stats["Structure_Stress"]),
            "avg_forward_return_10d": float(current_stats["fwd_return"])
        },

        "most_likely_next_regime_cluster": int(most_likely_next),
        "transition_probability": float(next_probs_no_self.max()),

        "expected_changes_if_transition_occurs": {
            "trend_strength_change": float(change["Trend_Strength"]),
            "direction_bias_change": float(change["Direction_Bias_Score"]),
            "volatility_change": float(change["Volatility_Stress"]),
            "structure_change": float(change["Structure_Stress"]),
            "forward_return_change": float(change["fwd_return"])
        }
    }



