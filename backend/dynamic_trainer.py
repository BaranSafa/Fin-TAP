"""
dynamic_trainer.py  —  Fin-TAP Backend

V0.7 Düzeltmeleri:
  1. Feature seçim bug'ı düzeltildi: DEFAULT_GROUPS artık her zaman eklenmez,
     sadece kullanıcı seçimi (veya varsayılan) kullanılır.
  2. Rastgele fallback düzeltildi: tahmin hatasında random gürültü yerine
     son bilinen fiyat kullanılır.
  3. Clip limiti 0.10 → 0.15 (BTC gibi volatil varlıklar için daha gerçekçi).
  4. Model cache eklendi: aynı ticker+model+feature kombinasyonu 30 dk cache'lenir.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import traceback
import time
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
)
from sklearn.preprocessing import RobustScaler

try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False

try:
    from .data_manager import get_processed_data, FEATURE_GROUPS, DEFAULT_GROUPS
except ImportError:
    from data_manager import get_processed_data, FEATURE_GROUPS, DEFAULT_GROUPS


# ──────────────────────────────────────────────────────────────
#  MODEL CACHE — aynı parametrelerle tekrar sorgu gelirse hız
# ──────────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}   # {cache_key: {"model": ..., "sc": ..., "feat_set": ..., "at": float}}
_MODEL_CACHE_TTL = 1800   # 30 dakika


def _cache_key(ticker: str, model_type: str, groups: list) -> str:
    return f"{ticker}|{model_type}|{'_'.join(sorted(groups))}"


def _get_cached_model(key: str):
    entry = _MODEL_CACHE.get(key)
    if entry and (time.time() - entry["at"]) < _MODEL_CACHE_TTL:
        print(f"[trainer] Cache HIT: {key}")
        return entry
    return None


def _set_cached_model(key: str, model, sc, feat_set: list):
    _MODEL_CACHE[key] = {"model": model, "sc": sc, "feat_set": feat_set, "at": time.time()}
    # Cache boyutunu sınırla (max 20 model)
    if len(_MODEL_CACHE) > 20:
        oldest = min(_MODEL_CACHE, key=lambda k: _MODEL_CACHE[k]["at"])
        del _MODEL_CACHE[oldest]


# ──────────────────────────────────────────────────────────────
#  YARDIMCI: Geçmiş pencereden feature satırı hesapla
# ──────────────────────────────────────────────────────────────
def _compute_row(
    p: list, h: list, lo: list, v: list, feat_cols: list
) -> dict:
    n = len(p)

    def s(lst, w):
        w = min(w, len(lst))
        return float(np.mean(lst[-w:])) if w > 0 else float(lst[-1])

    def sd(lst, w):
        w = min(w, len(lst))
        a = lst[-w:]
        return float(np.std(a)) if len(a) > 1 else 0.0

    def ewm_v(lst, span):
        return float(pd.Series(lst).ewm(span=span, adjust=False).mean().iloc[-1])

    def div(a, b, d=0.0):
        return float(a / b) if b and b != 0 else d

    cur = float(p[-1])
    p1  = float(p[-2]) if n >= 2 else cur
    p2  = float(p[-3]) if n >= 3 else p1
    p5  = float(p[-6]) if n >= 6 else p1
    p10 = float(p[-11]) if n >= 11 else p1
    p20 = float(p[-21]) if n >= 21 else p1

    p3   = float(p[-4]) if n >= 4 else p2

    lr1  = np.log(cur/p1)  if p1  > 0 else 0.0
    lr2  = np.log(cur/p2)  if p2  > 0 else 0.0
    lr3  = np.log(cur/p3)  if p3  > 0 else 0.0
    lr5  = div(np.log(cur/p5)  if p5  > 0 else 0, 5)
    lr10 = div(np.log(cur/p10) if p10 > 0 else 0, 10)
    lr20 = div(np.log(cur/p20) if p20 > 0 else 0, 20)

    # RSI
    def rsi_fn(per):
        diffs = [p[-i]-p[-i-1] for i in range(1, min(per+2, n))]
        if not diffs: return 50.0
        g = np.mean([max(d, 0) for d in diffs])
        l = np.mean([max(-d, 0) for d in diffs])
        return 100 - 100/(1 + g/l) if l > 0 else 100.0

    r7  = rsi_fn(7)
    r14 = rsi_fn(14)
    r21 = rsi_fn(21)

    # MACD
    e12 = ewm_v(p, 12); e26 = ewm_v(p, 26)
    macd_v = div(e12-e26, cur)
    sig    = macd_v * 0.85
    macd_h = macd_v - sig

    # Bollinger
    s20 = s(p, 20); sd20 = sd(p, 20)
    bbu = s20 + 2*sd20; bbl = s20 - 2*sd20
    bb_p = div(cur-bbl, bbu-bbl, 0.5)
    bb_w = div(bbu-bbl, s20)

    # SMA/EMA uzaklıkları
    ds = {w: div(cur - s(p, w), s(p, w)) for w in [5, 10, 20, 50, 100]}
    de = {sp: div(cur - ewm_v(p, sp), ewm_v(p, sp)) for sp in [9, 21]}

    # Volatilite
    lrs  = [np.log(p[-i]/p[-i-1]) for i in range(1, min(22, n)) if p[-i-1] > 0]
    vol5  = float(np.std(lrs[-5:]))  if len(lrs) >= 5  else 0.0
    vol10 = float(np.std(lrs[-10:])) if len(lrs) >= 10 else 0.0
    vol20 = float(np.std(lrs[-20:])) if len(lrs) >= 20 else 0.0
    rvol  = vol20 * np.sqrt(252)
    vrat  = div(vol10, vol20, 1.0)

    # ATR
    hi = float(h[-1]); li = float(lo[-1])
    atr_v = [
        max(h[-i]-lo[-i], abs(h[-i]-p[-i-1]), abs(lo[-i]-p[-i-1]))
        for i in range(1, min(15, len(h)))
    ]
    atr14 = float(np.mean(atr_v)) if atr_v else 0.0
    atr_p = div(atr14, cur)
    atr_t = div(atr14, float(np.mean(atr_v[-14:])) if len(atr_v) >= 14 else atr14, 1.0)

    # Stoch
    hh14 = max(h[-14:]) if len(h) >= 14 else max(h)
    ll14 = min(lo[-14:]) if len(lo) >= 14 else min(lo)
    stk  = div(100*(cur-ll14), hh14-ll14, 50.0)
    willr = div(-100*(hh14-cur), hh14-ll14, -50.0)

    # CCI
    tp = (hi + li + cur) / 3
    tp_arr = [(h[-i]+lo[-i]+p[-i])/3 for i in range(1, min(21, n))]
    tp_ma  = float(np.mean(tp_arr)) if tp_arr else tp
    tp_md  = float(np.mean(np.abs(np.array(tp_arr) - tp_ma))) if tp_arr else 0.0
    cci    = max(-300.0, min(300.0, div(tp-tp_ma, 0.015*tp_md)))

    # ADX
    pdms = [max(h[-i]-h[-i-1], 0) for i in range(1, min(15, len(h)-1))]
    mdms = [max(lo[-i-1]-lo[-i], 0) for i in range(1, min(15, len(lo)-1))]
    adxp = div(100*np.mean(pdms), atr14) if (pdms and atr14 > 0) else 0.0
    adxm = div(100*np.mean(mdms), atr14) if (mdms and atr14 > 0) else 0.0

    # ROC
    def roc(w):
        ref = p[-w-1] if n > w else p[0]
        return div(cur-ref, ref) * 100

    # Hacim
    vc   = float(v[-1])
    vs   = s(v, 14)
    vs50 = s(v, 50)
    vr   = min(div(vc, vs, 1.0), 10.0)
    vtr  = min(div(vs, vs50, 1.0), 5.0)
    pvc  = max(-5.0, min(5.0, lr1 * vr))

    # Pattern
    hlp = div(hi-li, cur)
    oc  = div(cur - float(p[-1]), cur)

    # 52 haftalık
    hi52 = max(p[-252:]) if n >= 252 else max(p)
    lo52 = min(p[-252:]) if n >= 252 else min(p)

    # Eğim
    sm20c  = s(p, 20)
    sm20p  = float(np.mean(p[-25:-5])) if n >= 25 else sm20c
    sm50c  = s(p, 50)
    sm50p  = float(np.mean(p[-55:-5])) if n >= 55 else sm50c
    sl20   = div(sm20c-sm20p, sm20p)
    sl50   = div(sm50c-sm50p, sm50p)

    row = {
        "lr_1":lr1, "lr_2":lr2, "lr_3":lr3, "lr_5":lr5, "lr_10":lr10, "lr_20":lr20,
        "rsi_7":r7, "rsi_14":r14, "rsi_21":r21, "rsi_diff":r14-r7, "rsi_mom":r14-r21,
        "macd":macd_v, "macd_sig":sig, "macd_hist":macd_h, "macd_mom":macd_v*0.05,
        "bb_pct":bb_p, "bb_width":bb_w,
        "dist_sma5":ds[5], "dist_sma10":ds[10], "dist_sma20":ds[20],
        "dist_sma50":ds[50], "dist_sma100":ds[100],
        "dist_ema9":de[9], "dist_ema21":de[21],
        "vol_5d":vol5, "vol_10d":vol10, "vol_20d":vol20, "rvol_20":rvol, "vol_ratio":vrat,
        "atr_pct":atr_p, "atr_trend":atr_t,
        "stoch_k":stk, "stoch_d":stk, "stoch_diff":0.0,
        "willr":willr, "cci":cci,
        "adx_plus":adxp, "adx_minus":adxm, "adx_diff":adxp-adxm,
        "roc_3":roc(3), "roc_5":roc(5), "roc_10":roc(10), "roc_20":roc(20),
        "v_ratio":vr, "v_trend":vtr, "pv_corr":pvc,
        "hl_pct":hlp, "open_close":oc,
        "dist_52w_high":div(cur-hi52, cur), "dist_52w_low":div(cur-lo52, cur),
        "sma20_slope":sl20, "sma50_slope":sl50,
    }
    return {k: row.get(k, 0.0) for k in feat_cols}


# ──────────────────────────────────────────────────────────────
#  ANA FONKSİYON
# ──────────────────────────────────────────────────────────────
def train_and_predict_dynamic(
    ticker: str,
    model_type: str,
    selected_feature_groups: list,
) -> tuple:
    """
    Returns: (future_prices: list, chart_data: dict) | (None, None)
    """
    # 1. Veri
    df = get_processed_data(ticker)
    if df is None or df.empty:
        print(f"[trainer] {ticker}: veri alınamadı")
        return None, None

    # 2. Feature seçimi — SADECE kullanıcı seçimi (veya varsayılan), her ikisi birden değil
    groups   = selected_feature_groups if selected_feature_groups else DEFAULT_GROUPS
    feat_set = []
    for g in groups:
        for col in FEATURE_GROUPS.get(g, []):
            if col in df.columns and col not in feat_set:
                feat_set.append(col)
    feat_set = [f for f in feat_set if f in df.columns]

    print(f"[trainer] {ticker} | {model_type} | {len(feat_set)} feat | {len(df)} satır")

    if len(feat_set) < 3:
        print("[trainer] yetersiz feature — DEFAULT_GROUPS kullanılıyor")
        feat_set = []
        for g in DEFAULT_GROUPS:
            for col in FEATURE_GROUPS.get(g, []):
                if col in df.columns and col not in feat_set:
                    feat_set.append(col)
        if len(feat_set) < 3:
            return None, None

    # 3. X, y
    X_df   = df[feat_set].copy()
    y_s    = df["target_lr"].copy()
    valid  = X_df.notna().all(axis=1) & y_s.notna()
    X_df   = X_df[valid]
    y_s    = y_s[valid]
    closes = df.loc[valid, "Close"]

    if len(X_df) < 100:
        print(f"[trainer] az satır: {len(X_df)}")
        return None, None

    X    = X_df.values.astype(float)
    y    = y_s.values.astype(float)

    # 4. Split
    split      = int(len(X) * 0.90)
    close_te   = closes.values[split:]

    # 5. Model cache kontrolü
    c_key    = _cache_key(ticker, model_type, groups)
    cached   = _get_cached_model(c_key)
    is_lstm  = model_type == "LSTM"

    if cached:
        model    = cached["model"]
        sc       = cached["sc"]
        feat_set = cached["feat_set"]
        X_sc     = sc.transform(X)
        X_tr, y_tr = X_sc[:split], y[:split]
        X_te, y_te = X_sc[split:], y[split:]
    else:
        sc       = RobustScaler()
        X_sc     = sc.fit_transform(X)
        X_tr, y_tr = X_sc[:split], y[:split]
        X_te, y_te = X_sc[split:], y[split:]
        model    = None

        # 6. Eğit
        try:
            if model_type == "LINEAR":
                model = Ridge(alpha=2.0)
                model.fit(X_tr, y_tr)

            elif model_type == "RANDOM_FOREST":
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=8, min_samples_leaf=5,
                    n_jobs=1, random_state=42
                )
                model.fit(X_tr, y_tr)

            elif model_type == "EXTRA_TREES":
                model = ExtraTreesRegressor(
                    n_estimators=100, max_depth=8, min_samples_leaf=5,
                    n_jobs=1, random_state=42
                )
                model.fit(X_tr, y_tr)

            elif model_type == "GRADIENT_BOOST":
                model = GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.05, max_depth=4,
                    subsample=0.8, min_samples_leaf=5, random_state=42
                )
                model.fit(X_tr, y_tr)

            elif model_type == "XGBOOST":
                if not HAS_XGB:
                    raise ImportError("xgboost kurulu değil — requirements.txt'e ekle")
                model = xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                    random_state=42, verbosity=0, n_jobs=1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

            elif model_type == "LIGHTGBM":
                if not HAS_LGB:
                    raise ImportError("lightgbm kurulu değil — requirements.txt'e ekle")
                model = lgb.LGBMRegressor(
                    n_estimators=200, learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, reg_lambda=2.0,
                    random_state=42, verbose=-1, n_jobs=1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])

            elif model_type == "LSTM":
                if not HAS_TF:
                    raise ImportError("tensorflow kurulu değil — Render free 512MB'a sığmayabilir")
                is_lstm = True
                seq_len = 15
                nf      = X_sc.shape[1]

                def mk_seq(Xd, yd, seq):
                    Xs, ys = [], []
                    for i in range(seq, len(Xd)):
                        Xs.append(Xd[i-seq:i]); ys.append(yd[i])
                    return np.array(Xs), np.array(ys)

                Xs, ys = mk_seq(X_sc, y, seq_len)
                sp2     = int(len(Xs) * 0.90)
                model = Sequential([
                    Input(shape=(seq_len, nf)),
                    LSTM(64, return_sequences=True),
                    Dropout(0.2),
                    LSTM(32),
                    Dense(1),
                ])
                model.compile(optimizer="adam", loss="huber")
                model.fit(
                    Xs[:sp2], ys[:sp2],
                    validation_data=(Xs[sp2:], ys[sp2:]),
                    epochs=30, batch_size=32,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                    verbose=0,
                )

            else:
                print(f"[trainer] bilinmeyen model: {model_type}")
                return None, None

        except Exception as e:
            print(f"[trainer] EĞİTİM HATASI ({model_type}): {e}")
            traceback.print_exc()
            return None, None

        # Modeli cache'e kaydet (LSTM hariç — çok büyük)
        if not is_lstm:
            _set_cached_model(c_key, model, sc, feat_set)

    # 7. Backtest
    try:
        if is_lstm:
            seq_len = 15
            Xs_all = []
            for i in range(seq_len, len(X_sc)):
                Xs_all.append(X_sc[i-seq_len:i])
            Xs_te_arr  = np.array(Xs_all[split-seq_len:])
            bt_ret     = model.predict(Xs_te_arr, verbose=0).flatten()
            bt_base    = closes.values[split: split+len(bt_ret)]
        else:
            bt_ret  = model.predict(X_te)
            bt_base = close_te

        n_bt     = min(len(bt_ret), len(bt_base)-1)
        bt_pred  = [
            float(bt_base[i]) * float(np.exp(np.clip(bt_ret[i], -0.15, 0.15)))
            for i in range(n_bt)
        ]
        bt_actual = [float(bt_base[i+1]) for i in range(n_bt)]
        bt_dates  = [d.strftime("%Y-%m-%d") for d in closes.index[split: split+n_bt]]

    except Exception as e:
        print(f"[trainer] backtest hatası: {e}")
        traceback.print_exc()
        bt_pred=[]; bt_actual=[]; bt_dates=[]

    # 8. Gelecek tahmin — rolling window
    ph = list(df["Close"].values[-80:].astype(float))
    hh = list(df["High"].values[-80:].astype(float))
    lh = list(df["Low"].values[-80:].astype(float))
    vh = list(df["Volume"].values[-80:].astype(float))

    last   = ph[-1]
    future = []

    if is_lstm:
        seq_len = 15
        seq_rows = []
        for i in range(seq_len):
            idx  = -(seq_len - i)
            ph_i = ph[:len(ph)+idx+1]
            if len(ph_i) < 2:
                ph_i = [ph[0]] * 2 + ph_i
            row = _compute_row(ph_i, hh[:len(ph_i)], lh[:len(ph_i)], vh[:len(ph_i)], feat_set)
            rdf = pd.DataFrame([row])[feat_set]
            seq_rows.append(sc.transform(rdf.values)[0])
        seq_arr = np.array(seq_rows)

        for _ in range(14):
            inp   = seq_arr[-seq_len:][np.newaxis]
            lr_p  = float(np.clip(model.predict(inp, verbose=0).flatten()[0], -0.15, 0.15))
            nxt   = last * np.exp(lr_p)
            future.append(nxt)
            ph.append(nxt); hh.append(nxt*1.005); lh.append(nxt*0.995)
            vh.append(float(np.mean(vh[-5:]))); last = nxt
            row = _compute_row(ph, hh, lh, vh, feat_set)
            rdf = pd.DataFrame([row])[feat_set]
            seq_arr = np.vstack([seq_arr, sc.transform(rdf.values)[0]])
    else:
        for step in range(14):
            try:
                row  = _compute_row(ph, hh, lh, vh, feat_set)
                rdf  = pd.DataFrame([row])[feat_set]
                lr_p = float(np.clip(model.predict(sc.transform(rdf.values))[0], -0.15, 0.15))
                nxt  = last * np.exp(lr_p)
            except Exception as e:
                print(f"[trainer] gelecek adım {step}: {e}")
                # Düzeltildi: rastgele gürültü yerine son fiyatı kullan
                nxt = last
            future.append(nxt)
            ph.append(nxt); hh.append(nxt*1.005); lh.append(nxt*0.995)
            vh.append(float(np.mean(vh[-5:]))); last = nxt

    if not future:
        print("[trainer] gelecek tahmin üretilemedi")
        return None, None

    # 9. Chart verisi
    last_date    = df.index[-1]
    future_dates = [
        (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, 15)
    ]
    chart = {
        "dates":            bt_dates + future_dates,
        "actual_prices":    bt_actual + [None]*14,
        "predicted_prices": bt_pred   + future,
    }
    return future, chart
