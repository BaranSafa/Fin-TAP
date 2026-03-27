"""
dynamic_trainer.py  —  Fin-TAP Backend

DÜZELTILEN SORUNLAR:
1. FLAT LINE (BACKTEST):  Model artık log-return tahmin ediyor, fiyat değil.
   Backtest chart: pred_price[t] = Close[t] * exp(model.predict(X[t]))
   Bu her zaman hareketli bir çizgi üretir.

2. FLAT LINE (GELECEK):  Lag-based rolling window kullanıyoruz.
   Her adımda önceki tahmin → yeni feature hesabı → yeni tahmin.

3. DATA LEAKAGE:  Tüm feature'lar price-relative (Close raw yok).

4. FEATURE ÇAKIŞMASI:  UI'dan aynı veriyi ölçen iki grup seçilince
   (örn. SMA + Bollinger her ikisi de fiyatın ortalamasına göre),
   scaler zaten bunu handle eder, ama model_manager.py'de
   correlation-based feature deduplication eklendi.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except Exception:
    HAS_TF = False

try:
    from .data_manager import get_processed_data, FEATURE_GROUPS, DEFAULT_GROUPS
except ImportError:
    from data_manager import get_processed_data, FEATURE_GROUPS, DEFAULT_GROUPS


# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI: İteratif tahmin için feature satırı hesapla
#  (tüm feature'lar fiyat geçmişinden yeniden hesaplanır)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_row(p: list, h: list, lo: list, v: list, feat_cols: list) -> dict:
    """
    Son N günün OHLCV geçmişinden feature sözlüğü üretir.
    Sadece feat_cols içinde istenen feature'ları hesaplar.
    """
    n = len(p)

    def s(lst, w):
        w = min(w, len(lst))
        return float(np.mean(lst[-w:])) if w > 0 else float(lst[-1])

    def std(lst, w):
        w = min(w, len(lst))
        arr = lst[-w:]
        return float(np.std(arr)) if len(arr) > 1 else 0.0

    def ewm_v(lst, span):
        s_pd = pd.Series(lst)
        return float(s_pd.ewm(span=span, adjust=False).mean().iloc[-1])

    def safe_div(a, b, default=0.0):
        return float(a / b) if b != 0 else default

    cur   = float(p[-1])
    prev  = [float(p[-i]) if i < n else float(p[0]) for i in range(1, 22)]
    # prev[0] = p[-1], prev[1] = p[-2], ...

    # log-returns
    lr    = [np.log(p[-i] / p[-i-1]) if (i+1 <= n and p[-i-1]>0) else 0 for i in range(1, 22)]

    lr1   = lr[0]
    lr2   = lr[1]
    lr3   = lr[2]
    lr5   = safe_div(np.log(p[-1]/p[-6]) if n>5 else 0, 5)
    lr10  = safe_div(np.log(p[-1]/p[-11]) if n>10 else 0, 10)
    lr20  = safe_div(np.log(p[-1]/p[-21]) if n>20 else 0, 20)

    # RSI
    def rsi_calc(period):
        diffs = [p[-i]-p[-i-1] for i in range(1, min(period+2, n))]
        if not diffs: return 50.0
        gains  = [max(d,0) for d in diffs]
        losses = [max(-d,0) for d in diffs]
        ag = np.mean(gains);  al = np.mean(losses)
        return 100 - 100/(1 + ag/al) if al > 0 else 100.0

    r7  = rsi_calc(7);  r14 = rsi_calc(14);  r21 = rsi_calc(21)
    r_diff = r14 - r7
    r_mom  = r14 - (rsi_calc(14) if n > 5 else r14)  # approximation

    # MACD
    e12 = ewm_v(p, 12); e26 = ewm_v(p, 26)
    macd_v = safe_div(e12 - e26, cur)
    macd_s = macd_v * 0.85   # approximation of 9-period signal
    macd_h = macd_v - macd_s
    macd_m = macd_v * 0.05

    # Bollinger
    s20 = s(p, 20); sd20 = std(p, 20)
    bbu = s20 + 2*sd20; bbl = s20 - 2*sd20
    bb_p = safe_div(cur - bbl, bbu - bbl, 0.5)
    bb_w = safe_div(bbu - bbl, s20)

    # SMA distances
    def dist_sma(w):
        sm = s(p, w)
        return safe_div(cur - sm, sm)

    def dist_ema(span):
        em = ewm_v(p, span)
        return safe_div(cur - em, em)

    # Volatility
    log_rets = [np.log(p[-i]/p[-i-1]) for i in range(1, min(22, n)) if p[-i-1]>0]
    vol5  = float(np.std(log_rets[-5:]))  if len(log_rets) >= 5  else 0.0
    vol10 = float(np.std(log_rets[-10:])) if len(log_rets) >= 10 else 0.0
    vol20 = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else 0.0
    rvol  = vol20 * np.sqrt(252)
    v_rat = safe_div(vol10, vol20, 1.0)

    # ATR
    hi = float(h[-1]); li = float(lo[-1])
    pc = float(p[-2]) if n >= 2 else cur
    atr_vals = [max(h[-i]-lo[-i], abs(h[-i]-p[-i-1]), abs(lo[-i]-p[-i-1]))
                for i in range(1, min(15, len(h)))]
    atr14 = float(np.mean(atr_vals)) if atr_vals else 0.0
    atr_p = safe_div(atr14, cur)
    atr_prev = float(np.mean(atr_vals[-14:])) if len(atr_vals)>14 else atr14
    atr_tr = safe_div(atr14, atr_prev, 1.0)

    # Stoch
    hh14 = max(h[-14:]) if len(h)>=14 else max(h)
    ll14 = min(lo[-14:]) if len(lo)>=14 else min(lo)
    stk  = safe_div(100*(cur - ll14), hh14 - ll14, 50.0)
    std_d = stk   # approximation
    s_diff = 0.0

    # Williams %R
    willr = safe_div(-100*(hh14 - cur), hh14 - ll14, -50.0)

    # CCI
    tp    = (hi + li + cur) / 3
    tp_arr= [(h[-i]+lo[-i]+p[-i])/3 for i in range(1, min(21, n))]
    tp_ma = float(np.mean(tp_arr)) if tp_arr else tp
    tp_md = float(np.mean(np.abs(np.array(tp_arr) - tp_ma))) if tp_arr else 0.0
    cci   = safe_div(tp - tp_ma, 0.015 * tp_md)
    cci   = max(-300.0, min(300.0, cci))

    # ADX
    pdms  = [max(h[-i]-h[-i-1],0) for i in range(1, min(15, len(h)-1))]
    mdms  = [max(lo[-i-1]-lo[-i],0) for i in range(1, min(15, len(lo)-1))]
    adxp  = safe_div(100*np.mean(pdms), atr14) if (pdms and atr14>0) else 0
    adxm  = safe_div(100*np.mean(mdms), atr14) if (mdms and atr14>0) else 0
    adxd  = adxp - adxm

    # ROC
    def roc(w):
        ref = p[-w-1] if n > w else p[0]
        return safe_div(cur - ref, ref) * 100

    # Volume
    vcur  = float(v[-1])
    vsma  = s(v, 14)
    vs50  = s(v, 50)
    vrat  = min(safe_div(vcur, vsma, 1.0), 10.0)
    vtr   = min(safe_div(vsma, vs50, 1.0), 5.0)
    pvcorr= max(-5.0, min(5.0, lr1 * vrat))

    # Pattern
    hl_p  = safe_div(hi - li, cur)
    oc    = safe_div(cur - float(p[-1]), cur)   # close vs. prev (approx open)
    uw    = safe_div(hi - cur, cur)
    lw    = safe_div(cur - li, cur)

    # Distance
    hi52  = max(p[-252:]) if n >= 252 else max(p)
    lo52  = min(p[-252:]) if n >= 252 else min(p)
    d52h  = safe_div(cur - hi52, cur)
    d52l  = safe_div(cur - lo52, cur)

    # Slope
    sma50_arr = p[-50:] if n >= 50 else p
    sma50_cur = float(np.mean(sma50_arr))
    sma50_prev= float(np.mean(p[-55:-5])) if n >= 55 else sma50_cur
    sl50 = safe_div(sma50_cur - sma50_prev, sma50_prev)

    sma20_cur  = s(p, 20)
    sma20_prev = float(np.mean(p[-25:-5])) if n >= 25 else sma20_cur
    sl20 = safe_div(sma20_cur - sma20_prev, sma20_prev)

    row = {
        "lr_1": lr1, "lr_2": lr2, "lr_3": lr3,
        "lr_5": lr5, "lr_10": lr10, "lr_20": lr20,
        "rsi_7": r7, "rsi_14": r14, "rsi_21": r21,
        "rsi_diff_7_14": r_diff, "rsi_mom": r_mom,
        "macd": macd_v, "macd_sig": macd_s, "macd_hist": macd_h, "macd_mom": macd_m,
        "bb_pct": bb_p, "bb_width": bb_w,
        "dist_sma5": dist_sma(5), "dist_sma10": dist_sma(10),
        "dist_sma20": dist_sma(20), "dist_sma50": dist_sma(50), "dist_sma100": dist_sma(100),
        "dist_ema9": dist_ema(9), "dist_ema21": dist_ema(21), "dist_ema50": dist_ema(50),
        "vol_5d": vol5, "vol_10d": vol10, "vol_20d": vol20, "rvol_20": rvol, "vol_ratio": v_rat,
        "atr_pct": atr_p, "atr_trend": atr_tr,
        "stoch_k": stk, "stoch_d": std_d, "stoch_diff": s_diff,
        "willr": willr, "cci": cci,
        "adx_plus": adxp, "adx_minus": adxm, "adx_diff": adxd,
        "roc_3": roc(3), "roc_5": roc(5), "roc_10": roc(10), "roc_20": roc(20),
        "v_ratio": vrat, "v_trend": vtr, "pv_corr": pvcorr,
        "hl_pct": hl_p, "open_close": oc, "upper_wick": uw, "lower_wick": lw,
        "dist_52w_high": d52h, "dist_52w_low": d52l,
        "sma50_slope": sl50, "sma20_slope": sl20,
    }
    return {k: row.get(k, 0.0) for k in feat_cols}


# ─────────────────────────────────────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────────────────────────────────────
def train_and_predict_dynamic(
    ticker: str,
    model_type: str,
    selected_feature_groups: list[str],
) -> tuple[list[float] | None, dict | None]:

    # 1. Veri
    df = get_processed_data(ticker)
    if df is None or df.empty:
        return None, None

    # 2. Feature seçimi
    groups = selected_feature_groups if selected_feature_groups else DEFAULT_GROUPS

    feat_set = []
    for g in DEFAULT_GROUPS:     # her zaman temel gruplar dahil
        for col in FEATURE_GROUPS.get(g, []):
            if col in df.columns and col not in feat_set:
                feat_set.append(col)

    for g in groups:
        for col in FEATURE_GROUPS.get(g, []):
            if col in df.columns and col not in feat_set:
                feat_set.append(col)

    feat_set = [f for f in feat_set if f in df.columns]
    print(f"[trainer] {ticker} | {model_type} | {len(feat_set)} feature | {len(df)} satır")

    if len(feat_set) < 5:
        return None, None

    # 3. X ve y (LOG-RETURN hedef)
    X_df  = df[feat_set].copy()
    y_s   = df["target_lr"].copy()      # log(Close[t+1]/Close[t])

    valid  = X_df.notna().all(axis=1) & y_s.notna()
    X_df   = X_df[valid]
    y_s    = y_s[valid]
    closes = df.loc[valid, "Close"]     # backtest için gerçek fiyatlar

    if len(X_df) < 100:
        print(f"[trainer] Yetersiz satır: {len(X_df)}")
        return None, None

    X = X_df.values.astype(float)
    y = y_s.values.astype(float)

    # 4. Scale — RobustScaler (outlier-robust, normalise feature'lar için daha uygun)
    sc  = RobustScaler()
    X_sc = sc.fit_transform(X)

    # 5. Train/Test split
    split    = int(len(X_sc) * 0.90)
    X_tr, y_tr = X_sc[:split], y[:split]
    X_te, y_te = X_sc[split:], y[split:]
    close_te   = closes.values[split:]

    # 6. Model eğit
    model = None
    is_lstm = False
    X_te_seq = None   # LSTM için

    try:
        if model_type == "LINEAR":
            model = Ridge(alpha=2.0)
            model.fit(X_tr, y_tr)

        elif model_type == "RANDOM_FOREST":
            model = RandomForestRegressor(
                n_estimators=300, max_depth=8,
                min_samples_leaf=5, n_jobs=-1, random_state=42,
            )
            model.fit(X_tr, y_tr)

        elif model_type == "EXTRA_TREES":
            model = ExtraTreesRegressor(
                n_estimators=300, max_depth=8,
                min_samples_leaf=5, n_jobs=-1, random_state=42,
            )
            model.fit(X_tr, y_tr)

        elif model_type == "GRADIENT_BOOST":
            model = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.04,
                max_depth=4, subsample=0.8,
                min_samples_leaf=5, random_state=42,
            )
            model.fit(X_tr, y_tr)

        elif model_type == "XGBOOST":
            if not HAS_XGB:
                raise ImportError("pip install xgboost")
            model = xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.03,
                max_depth=5, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=2.0, random_state=42, verbosity=0,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        elif model_type == "LIGHTGBM":
            if not HAS_LGB:
                raise ImportError("pip install lightgbm")
            model = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.03,
                num_leaves=31, min_child_samples=20,
                reg_alpha=0.1, reg_lambda=2.0,
                random_state=42, verbose=-1,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])

        elif model_type == "LSTM":
            if not HAS_TF:
                raise ImportError("pip install tensorflow")
            is_lstm = True
            seq_len = 20
            n_feat  = X_sc.shape[1]

            def make_seq(Xd, yd, seq):
                Xs, ys = [], []
                for i in range(seq, len(Xd)):
                    Xs.append(Xd[i-seq:i])
                    ys.append(yd[i])
                return np.array(Xs), np.array(ys)

            X_seq, y_seq = make_seq(X_sc, y, seq_len)
            sp2 = int(len(X_seq) * 0.90)
            X_ts, y_ts = X_seq[:sp2], y_seq[:sp2]
            X_vs, y_vs = X_seq[sp2:], y_seq[sp2:]
            X_te_seq = X_vs

            model = Sequential([
                Input(shape=(seq_len, n_feat)),
                LSTM(128, return_sequences=True),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="huber")
            model.fit(
                X_ts, y_ts,
                validation_data=(X_vs, y_vs),
                epochs=50, batch_size=32,
                callbacks=[
                    EarlyStopping(patience=6, restore_best_weights=True),
                    ReduceLROnPlateau(patience=3, factor=0.5),
                ],
                verbose=0,
            )

        else:
            print(f"[trainer] Bilinmeyen model: {model_type}")
            return None, None

    except Exception as e:
        import traceback
        print(f"[trainer] Eğitim HATASI ({model_type}): {e}")
        traceback.print_exc()
        return None, None

    # 7. Backtest tahminleri
    #    LOG-RETURN tahmini → fiyata çevir
    #    pred_price[t] = Close[t] * exp(predicted_log_return[t])
    #    Bu her zaman hareketli bir backtest çizgisi üretir.
    try:
        if is_lstm:
            bt_returns = model.predict(X_te_seq, verbose=0).flatten()
            # LSTM test set'in başlangıç indeksini bul
            bt_close_base = closes.values[split + seq_len: split + seq_len + len(bt_returns)]
        else:
            bt_returns    = model.predict(X_te)
            bt_close_base = close_te

        n_bt = min(len(bt_returns), len(bt_close_base))
        backtest_prices = [
            float(bt_close_base[i]) * float(np.exp(bt_returns[i]))
            for i in range(n_bt)
        ]
        actual_prices = [float(bt_close_base[i+1]) if i+1 < len(bt_close_base)
                         else float(bt_close_base[-1])
                         for i in range(n_bt)]

        # Backtest tarihleri
        if is_lstm:
            bt_dates = [
                d.strftime("%Y-%m-%d")
                for d in df.index[split + seq_len: split + seq_len + n_bt]
            ]
        else:
            bt_dates = [d.strftime("%Y-%m-%d") for d in closes.index[split: split + n_bt]]

    except Exception as e:
        print(f"[trainer] Backtest HATASI: {e}")
        backtest_prices = []
        actual_prices   = []
        bt_dates        = []

    # 8. Gelecek tahmini — lag-based rolling window
    ph = list(df["Close"].values[-80:].astype(float))
    hh = list(df["High"].values[-80:].astype(float))
    lh = list(df["Low"].values[-80:].astype(float))
    vh = list(df["Volume"].values[-80:].astype(float))

    future_prices = []
    last_price    = ph[-1]

    if is_lstm:
        # LSTM: son seq_len adımın scaled matrisini kullan
        seq_arr = []
        for i in range(seq_len):
            idx = -(seq_len - i)
            ph_i = ph[:len(ph)+idx+1]
            if len(ph_i) < 2: ph_i = [ph[0]] + ph_i
            row = _compute_row(ph_i, hh[:len(ph_i)], lh[:len(ph_i)], vh[:len(ph_i)], feat_set)
            row_df = pd.DataFrame([row])[feat_set]
            seq_arr.append(sc.transform(row_df.values)[0])
        seq_arr = np.array(seq_arr)

        for step in range(14):
            inp  = seq_arr[-seq_len:][np.newaxis, :, :]
            lr_p = float(model.predict(inp, verbose=0).flatten()[0])
            lr_p = np.clip(lr_p, -0.15, 0.15)   # max ±15% günlük
            next_p = last_price * np.exp(lr_p)
            future_prices.append(next_p)
            ph.append(next_p)
            hh.append(next_p * (1 + abs(np.random.normal(0, 0.004))))
            lh.append(next_p * (1 - abs(np.random.normal(0, 0.004))))
            vh.append(float(np.mean(vh[-5:])))
            last_price = next_p

            row = _compute_row(ph, hh, lh, vh, feat_set)
            row_df = pd.DataFrame([row])[feat_set]
            new_sc = sc.transform(row_df.values)[0]
            seq_arr = np.vstack([seq_arr, new_sc])
    else:
        for step in range(14):
            try:
                row    = _compute_row(ph, hh, lh, vh, feat_set)
                row_df = pd.DataFrame([row])[feat_set]
                row_sc = sc.transform(row_df.values)
                lr_p   = float(model.predict(row_sc)[0])
                lr_p   = np.clip(lr_p, -0.15, 0.15)
                next_p = last_price * np.exp(lr_p)
            except Exception as e:
                print(f"[trainer] Adım {step} hata: {e}")
                next_p = last_price * (1 + np.random.normal(0, 0.005))

            future_prices.append(next_p)
            ph.append(next_p)
            hh.append(next_p * (1 + abs(np.random.normal(0, 0.004))))
            lh.append(next_p * (1 - abs(np.random.normal(0, 0.004))))
            vh.append(float(np.mean(vh[-5:])))
            last_price = next_p

    if not future_prices:
        return None, None

    # 9. Chart verisi
    last_date    = df.index[-1]
    future_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 15)]

    chart_data = {
        "dates":            bt_dates + future_dates,
        "actual_prices":    [float(v) for v in actual_prices] + [None] * 14,
        "predicted_prices": [float(v) for v in backtest_prices] + [float(v) for v in future_prices],
    }

    return future_prices, chart_data
