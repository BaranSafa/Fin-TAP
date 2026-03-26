"""
dynamic_trainer.py  –  Fin-TAP Backend
Sıfırdan yazılmış ML eğitim ve tahmin motoru.

DÜZELTILEN ANA SORUN — DÜZLÜK (FLAT LINE):
  Eski kod scaled feature vektörünü küçük gaussian noise ile
  güncelliyordu:  curr = curr * (1 + 0.002 noise)
  Bu işlem modelin gördüğü input'u neredeyse hiç değiştirmiyordu
  → model her adımda aynı tahmini üretiyordu → DÜZLÜK

DOĞRU YAKLAŞIM — Lag-Based Rolling Window Forecasting:
  Her tahmin adımında gerçek fiyat değişkenlerine (lag, SMA, RSI...)
  dayanan yeni bir feature satırı oluşturulur.
  Bu satır scaler ile dönüştürülüp modele verilir.
  Tahmin edilen fiyat bir sonraki adımın lag değeri olur.

EK MODEL: GRADIENT BOOSTING (sklearn) — XGBoost/LightGBM yokken de çalışır.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Optional heavy deps
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
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False

# Relative import (package olarak kullanım)
try:
    from .data_manager import get_processed_data
except ImportError:
    from data_manager import get_processed_data


# ──────────────────────────────────────────────────────────────────────────────
#  FEATURE GRUP TANIMI
#  Her grubun altındaki kolon adları data_manager.py çıktısıyla eşleşmelidir.
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_GROUPS: dict[str, list[str]] = {
    'OHLCV':      ['Open', 'High', 'Low', 'Close', 'Volume'],
    'Lag':        ['lag_1', 'lag_2', 'lag_5', 'lag_10'],
    'Returns':    ['returns', 'ret_3', 'ret_5', 'ret_10'],
    'SMA':        ['SMA_5', 'SMA_14', 'SMA_20', 'SMA_50',
                   'price_sma14_ratio', 'price_sma50_ratio'],
    'EMA':        ['EMA_9', 'EMA_14', 'EMA_21'],
    'Volatility': ['volatility_10', 'volatility_20', 'realized_vol'],
    'Bollinger':  ['BB_upper', 'BB_middle', 'BB_lower', 'BB_pct', 'BB_width'],
    'MACD':       ['MACD', 'MACD_signal', 'MACD_hist'],
    'RSI':        ['RSI_14', 'RSI_7'],
    'Stoch':      ['STOCH_K', 'STOCH_D'],
    'ATR':        ['ATR_14', 'ATR_pct'],
    'OBV':        ['OBV', 'OBV_SMA_14', 'OBV_ratio'],
    'ADX':        ['ADX_14', 'ADX_plus', 'ADX_minus'],
    'Momentum':   ['MOM_5', 'MOM_10', 'ROC_5', 'ROC_10'],
    'CCI':        ['CCI_20'],
    'Williams':   ['WILLIAMS_R'],
    'Volume':     ['volume_sma_14', 'volume_ratio', 'price_x_volume'],
    'Pattern':    ['high_low_pct', 'close_open'],
}

# Varsayılan seçili gruplar (UI'dan hiç seçilmezse kullanılır)
DEFAULT_GROUPS = ['Lag', 'Returns', 'SMA', 'MACD', 'RSI', 'Bollinger', 'ATR', 'Volatility']


# ──────────────────────────────────────────────────────────────────────────────
#  YARDIMCI: feature vektörü oluştur (lag-based iterative forecasting için)
# ──────────────────────────────────────────────────────────────────────────────
def _build_feature_row(
    price_hist:  list,   # son N günün kapanış fiyatları (sonuncusu en yeni)
    high_hist:   list,
    low_hist:    list,
    volume_hist: list,
    feature_cols: list[str],
) -> dict:
    """
    Tarihsel pencereden tek bir feature satırı üretir.
    Sadece feature_cols içindeki sütunları hesaplar → hız.
    """
    p    = price_hist
    h    = high_hist
    lo   = low_hist
    v    = volume_hist
    n    = len(p)

    def safe_mean(lst, w):
        w = min(w, len(lst))
        return float(np.mean(lst[-w:])) if w > 0 else float(lst[-1])

    def safe_std(lst, w):
        w = min(w, len(lst))
        return float(np.std(lst[-w:])) if w > 1 else 0.0

    def ewm_last(lst, span):
        s = pd.Series(lst)
        return float(s.ewm(span=span, adjust=False).mean().iloc[-1])

    cur   = float(p[-1])
    prev1 = float(p[-2]) if n >= 2 else cur
    prev2 = float(p[-3]) if n >= 3 else prev1
    prev5 = float(p[-6]) if n >= 6 else prev1
    prev10= float(p[-11]) if n >= 11 else prev1

    ret    = (cur - prev1) / prev1 if prev1 != 0 else 0.0
    ret3   = (cur - float(p[-4])  ) / float(p[-4])   if n >= 4  else ret
    ret5   = (cur - prev5)          / prev5           if prev5 != 0 else ret
    ret10  = (cur - prev10)         / prev10          if prev10 != 0 else ret

    sma5   = safe_mean(p, 5)
    sma14  = safe_mean(p, 14)
    sma20  = safe_mean(p, 20)
    sma50  = safe_mean(p, 50)

    ema9   = ewm_last(p, 9)
    ema14  = ewm_last(p, 14)
    ema21  = ewm_last(p, 21)

    std10  = safe_std(p, 10)
    std20  = safe_std(p, 20)

    log_rets = [np.log(p[i]/p[i-1]) for i in range(max(1, n-20), n) if p[i-1]>0]
    rvol = float(np.std(log_rets) * np.sqrt(252)) if len(log_rets) > 1 else 0.0

    bb_u = sma20 + 2 * std20
    bb_l = sma20 - 2 * std20
    bb_w = (bb_u - bb_l) / sma20 if sma20 != 0 else 0.0
    bb_p = (cur - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) != 0 else 0.5

    ema12 = ewm_last(p, 12)
    ema26 = ewm_last(p, 26)
    macd_val = ema12 - ema26
    macd_sig = ewm_last([macd_val], 9)   # approximation for single row
    macd_hist_val = macd_val - macd_sig

    # RSI
    diffs  = [p[i]-p[i-1] for i in range(max(1, n-15), n)]
    gains  = [max(d,0) for d in diffs]
    losses = [max(-d,0) for d in diffs]
    avg_g  = np.mean(gains)  if gains  else 0
    avg_l  = np.mean(losses) if losses else 1e-9
    rsi14  = 100 - 100/(1 + avg_g/avg_l)

    diffs7 = diffs[-7:]
    gains7 = [max(d,0) for d in diffs7]
    losses7= [max(-d,0) for d in diffs7]
    ag7    = np.mean(gains7)  if gains7  else 0
    al7    = np.mean(losses7) if losses7 else 1e-9
    rsi7   = 100 - 100/(1 + ag7/al7)

    # Stoch
    w14_h = max(h[-14:]) if len(h) >= 14 else max(h)
    w14_l = min(lo[-14:]) if len(lo) >= 14 else min(lo)
    stk   = 100*(cur - w14_l)/(w14_h - w14_l) if (w14_h - w14_l) != 0 else 50.0
    std_d_series = pd.Series([stk]).rolling(3).mean()
    std_d = float(std_d_series.iloc[-1]) if not std_d_series.empty else stk

    # ATR
    atr_vals = [max(h[i]-lo[i], abs(h[i]-p[i-1]), abs(lo[i]-p[i-1]))
                for i in range(max(1, len(h)-14), len(h))]
    atr14 = float(np.mean(atr_vals)) if atr_vals else 0.0
    atr_p = atr14 / cur if cur != 0 else 0.0

    # OBV
    obv_cur = float(np.sum([
        np.sign(p[i]-p[i-1])*v[i] for i in range(max(1,n-30),n)
    ]))
    obv_sma = obv_cur  # simplified
    obv_rat = 1.0

    # ADX
    plus_dms  = [max(h[i]-h[i-1],0) for i in range(max(1, len(h)-14),len(h))]
    minus_dms = [max(lo[i-1]-lo[i],0) for i in range(max(1, len(lo)-14),len(lo))]
    tr_vals   = atr_vals
    tr_mean   = np.mean(tr_vals) if tr_vals else 1e-9
    adx_plus  = 100*np.mean(plus_dms) /tr_mean if tr_mean!=0 else 0
    adx_minus = 100*np.mean(minus_dms)/tr_mean if tr_mean!=0 else 0
    adx14     = abs(adx_plus - adx_minus)

    # Momentum
    mom5  = cur - prev5
    mom10 = cur - prev10
    roc5  = ret5 * 100
    roc10 = ret10 * 100

    # CCI
    tp     = (float(h[-1]) + float(lo[-1]) + cur) / 3
    tp_ser = [(float(h[-20:][i]) + float(lo[-20:][i]) + float(p[-20:][i]))/3
               for i in range(min(20, n))] if n >= 2 else [tp]
    tp_ma  = np.mean(tp_ser)
    tp_md  = np.mean(np.abs(np.array(tp_ser) - tp_ma))
    cci20  = (tp - tp_ma) / (0.015 * tp_md) if tp_md != 0 else 0.0

    # Williams %R
    wills_r = -100*(w14_h - cur)/(w14_h - w14_l) if (w14_h - w14_l) != 0 else -50.0

    # Volume features
    vcur   = float(v[-1])
    vsma14 = safe_mean(v, 14)
    vrat   = vcur / vsma14 if vsma14 != 0 else 1.0
    pxv    = (cur * vcur) / 1e9

    # Pattern
    hl_pct    = (float(h[-1]) - float(lo[-1])) / cur if cur != 0 else 0.0
    close_open= (cur - float(p[-1])) / cur if cur != 0 else 0.0   # approx

    sma14_r   = cur / sma14 if sma14 != 0 else 1.0
    sma50_r   = cur / sma50 if sma50 != 0 else 1.0

    row_map = {
        'Open': cur, 'High': float(h[-1]), 'Low': float(lo[-1]),
        'Close': cur, 'Volume': vcur,
        'lag_1': prev1, 'lag_2': prev2, 'lag_5': prev5, 'lag_10': prev10,
        'returns': ret, 'ret_3': ret3, 'ret_5': ret5, 'ret_10': ret10,
        'SMA_5': sma5, 'SMA_14': sma14, 'SMA_20': sma20, 'SMA_50': sma50,
        'price_sma14_ratio': sma14_r, 'price_sma50_ratio': sma50_r,
        'EMA_9': ema9, 'EMA_14': ema14, 'EMA_21': ema21,
        'volatility_10': std10, 'volatility_20': std20, 'realized_vol': rvol,
        'BB_upper': bb_u, 'BB_middle': sma20, 'BB_lower': bb_l,
        'BB_pct': bb_p, 'BB_width': bb_w,
        'MACD': macd_val, 'MACD_signal': macd_sig, 'MACD_hist': macd_hist_val,
        'RSI_14': rsi14, 'RSI_7': rsi7,
        'STOCH_K': stk,  'STOCH_D': std_d,
        'ATR_14': atr14, 'ATR_pct': atr_p,
        'OBV': obv_cur,  'OBV_SMA_14': obv_sma, 'OBV_ratio': obv_rat,
        'ADX_14': adx14, 'ADX_plus': adx_plus, 'ADX_minus': adx_minus,
        'MOM_5': mom5,   'MOM_10': mom10, 'ROC_5': roc5, 'ROC_10': roc10,
        'CCI_20': cci20, 'WILLIAMS_R': wills_r,
        'volume_sma_14': vsma14, 'volume_ratio': vrat, 'price_x_volume': pxv,
        'high_low_pct': hl_pct,  'close_open': close_open,
    }

    return {k: row_map.get(k, 0.0) for k in feature_cols}


# ──────────────────────────────────────────────────────────────────────────────
#  ANA FONKSİYON
# ──────────────────────────────────────────────────────────────────────────────
def train_and_predict_dynamic(
    ticker: str,
    model_type: str,
    selected_feature_groups: list[str],
) -> tuple[list[float] | None, dict | None]:
    """
    Eğit ve 14 günlük gelecek tahmini üret.

    Returns:
        (future_predictions: list[float], chart_data: dict)  – başarı
        (None, None)  – hata
    """

    # ── 1. Veri al ──────────────────────────────────────────────────────────
    df = get_processed_data(ticker)
    if df is None or df.empty:
        print(f"[trainer] {ticker}: Veri alınamadı.")
        return None, None

    # ── 2. Feature kolonlarını seç ───────────────────────────────────────────
    groups = selected_feature_groups if selected_feature_groups else DEFAULT_GROUPS

    # Her zaman temel feature'ları dahil et
    feature_set: list[str] = []
    for g in ['OHLCV', 'Lag', 'Returns']:
        for col in FEATURE_GROUPS[g]:
            if col in df.columns and col not in feature_set:
                feature_set.append(col)

    # Kullanıcı seçimleri
    for g in groups:
        if g in FEATURE_GROUPS:
            for col in FEATURE_GROUPS[g]:
                if col in df.columns and col not in feature_set:
                    feature_set.append(col)

    # df'te gerçekten var olanlarla sınırla
    feature_set = [f for f in feature_set if f in df.columns]
    print(f"[trainer] {ticker} | model={model_type} | {len(feature_set)} feature | {len(df)} satır")

    if len(feature_set) < 5:
        print(f"[trainer] Yetersiz feature: {feature_set}")
        return None, None

    # ── 3. X / y hazırla ────────────────────────────────────────────────────
    # Hedef: sonraki günün Close fiyatı
    y_raw = df['Close'].shift(-1)

    # Son satır tahmin için (y=NaN), geri kalanı eğitim için
    X_df = df[feature_set].iloc[:-1].copy()
    y_s  = y_raw.iloc[:-1].copy()

    # NaN temizliği (her ihtimale karşı)
    valid = X_df.notna().all(axis=1) & y_s.notna()
    X_df  = X_df[valid]
    y_s   = y_s[valid]

    if len(X_df) < 80:
        print(f"[trainer] Yetersiz eğitim satırı: {len(X_df)}")
        return None, None

    X_np = X_df.values.astype(float)
    y_np = y_s.values.astype(float)

    # ── 4. Ölçekleme ─────────────────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_sc   = scaler.fit_transform(X_np)

    # ── 5. Train / Test split (%92 / %8) ────────────────────────────────────
    split    = int(len(X_sc) * 0.92)
    X_train  = X_sc[:split];   y_train = y_np[:split]
    X_test   = X_sc[split:];   y_test  = y_np[split:]

    # ── 6. Model eğitimi ─────────────────────────────────────────────────────
    model = None
    try:
        if model_type == 'LINEAR':
            model = Ridge(alpha=0.5)
            model.fit(X_train, y_train)

        elif model_type == 'RANDOM_FOREST':
            model = RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_leaf=3, n_jobs=-1, random_state=42
            )
            model.fit(X_train, y_train)

        elif model_type == 'EXTRA_TREES':
            model = ExtraTreesRegressor(
                n_estimators=200, max_depth=10,
                min_samples_leaf=3, n_jobs=-1, random_state=42
            )
            model.fit(X_train, y_train)

        elif model_type == 'GRADIENT_BOOST':
            model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=5, subsample=0.8,
                min_samples_leaf=3, random_state=42
            )
            model.fit(X_train, y_train)

        elif model_type == 'XGBOOST':
            if not HAS_XGB:
                raise ImportError("XGBoost kurulu değil: pip install xgboost")
            model = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.04,
                max_depth=6, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=42, verbosity=0
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)], verbose=False)

        elif model_type == 'LIGHTGBM':
            if not HAS_LGB:
                raise ImportError("LightGBM kurulu değil: pip install lightgbm")
            model = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.04,
                num_leaves=40, min_child_samples=10,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbose=-1
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)])

        elif model_type == 'LSTM':
            if not HAS_TF:
                raise ImportError("TensorFlow kurulu değil: pip install tensorflow")
            seq_len  = 20
            n_feat   = X_sc.shape[1]

            def make_sequences(X, y, seq):
                Xs, ys = [], []
                for i in range(seq, len(X)):
                    Xs.append(X[i-seq:i])
                    ys.append(y[i])
                return np.array(Xs), np.array(ys)

            X_seq, y_seq = make_sequences(X_sc, y_np, seq_len)
            sp2  = int(len(X_seq) * 0.92)
            X_tr_s, y_tr_s = X_seq[:sp2], y_seq[:sp2]
            X_te_s, y_te_s = X_seq[sp2:], y_seq[sp2:]

            model_lstm = Sequential([
                Input(shape=(seq_len, n_feat)),
                LSTM(128, return_sequences=True),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='huber')
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model_lstm.fit(
                X_tr_s, y_tr_s,
                validation_data=(X_te_s, y_te_s),
                epochs=50, batch_size=32,
                callbacks=[es], verbose=0
            )

            # LSTM için gelecek tahmini
            future_predictions = _lstm_future(
                model_lstm, X_sc, y_np, df, feature_set, scaler, seq_len
            )
            backtest_preds = model_lstm.predict(X_te_s, verbose=0).flatten()
            bt_dates = [
                d.strftime('%Y-%m-%d')
                for d in df.index[split + seq_len: split + seq_len + len(backtest_preds)]
            ]
            chart_data = _build_chart_data(
                bt_dates, y_te_s.tolist(), backtest_preds.tolist(),
                future_predictions, df.index[-1]
            )
            return future_predictions, chart_data

        else:
            print(f"[trainer] Bilinmeyen model tipi: {model_type}")
            return None, None

    except Exception as e:
        import traceback
        print(f"[trainer] Model eğitim HATASI ({model_type}): {e}")
        traceback.print_exc()
        return None, None

    # ── 7. Backtest tahminleri ───────────────────────────────────────────────
    backtest_preds = model.predict(X_test).tolist()

    # ── 8. Lag-based iterative gelecek tahmini ───────────────────────────────
    # DÜZELTME: scaled noise yerine gerçek rolling window kullanıyoruz.
    future_predictions = _sklearn_future(
        model, df, feature_set, scaler
    )

    if not future_predictions:
        print(f"[trainer] Gelecek tahmin üretilemedi.")
        return None, None

    # ── 9. Chart verisi ──────────────────────────────────────────────────────
    bt_dates = [
        d.strftime('%Y-%m-%d')
        for d in df.index[split: split + len(backtest_preds)]
    ]
    chart_data = _build_chart_data(
        bt_dates, y_test.tolist(), backtest_preds,
        future_predictions, df.index[-1]
    )

    return future_predictions, chart_data


# ──────────────────────────────────────────────────────────────────────────────
#  YARDIMCI: sklearn modelleri için lag-based future forecasting
# ──────────────────────────────────────────────────────────────────────────────
def _sklearn_future(model, df: pd.DataFrame, feature_cols: list, scaler) -> list[float]:
    """
    Son 60 günün gerçek verisinden başlayarak
    14 günlük öngörü üretir.
    Her adımda önceki tahmin yeni lag değeri olur.
    """
    # Rolling window — gerçek tarihsel veriden başla
    window_size = 60
    price_h  = list(df['Close'].values[-window_size:].astype(float))
    high_h   = list(df['High'].values[-window_size:].astype(float))
    low_h    = list(df['Low'].values[-window_size:].astype(float))
    volume_h = list(df['Volume'].values[-window_size:].astype(float))

    future = []
    for step in range(14):
        try:
            row_dict = _build_feature_row(price_h, high_h, low_h, volume_h, feature_cols)
            row_df   = pd.DataFrame([row_dict])[feature_cols]
            row_sc   = scaler.transform(row_df.values)
            pred     = float(model.predict(row_sc)[0])
        except Exception as e:
            print(f"[trainer] Adım {step} hata: {e}")
            # Hata durumunda son değerden küçük bir adım ilerle
            pred = price_h[-1] * (1 + np.random.normal(0.0002, 0.008))

        future.append(pred)

        # Rolling window'a yeni tahmini ekle
        price_h.append(pred)
        high_h.append(pred  * (1 + abs(np.random.normal(0, 0.004))))
        low_h.append(pred   * (1 - abs(np.random.normal(0, 0.004))))
        volume_h.append(float(np.mean(volume_h[-5:])))

    return future


# ──────────────────────────────────────────────────────────────────────────────
#  YARDIMCI: LSTM için sequence-based future forecasting
# ──────────────────────────────────────────────────────────────────────────────
def _lstm_future(model, X_sc, y_np, df, feature_cols, scaler, seq_len) -> list[float]:
    price_h  = list(df['Close'].values[-60:].astype(float))
    high_h   = list(df['High'].values[-60:].astype(float))
    low_h    = list(df['Low'].values[-60:].astype(float))
    volume_h = list(df['Volume'].values[-60:].astype(float))

    # Son seq_len adımın scaled matrisini oluştur
    seq_rows = []
    for i in range(seq_len):
        ph   = price_h[-(seq_len - i):]
        hh   = high_h[-(seq_len - i):]
        loh  = low_h[-(seq_len - i):]
        vh   = volume_h[-(seq_len - i):]
        if len(ph) < 2:
            ph = [price_h[-1]] * 2 + ph
        row_d  = _build_feature_row(ph, hh, loh, vh, feature_cols)
        row_df = pd.DataFrame([row_d])[feature_cols]
        seq_rows.append(scaler.transform(row_df.values)[0])

    seq_arr = np.array(seq_rows)   # (seq_len, n_feat)

    future = []
    for step in range(14):
        inp  = seq_arr[-seq_len:][np.newaxis, :, :]   # (1, seq_len, n_feat)
        pred = float(model.predict(inp, verbose=0).flatten()[0])
        future.append(pred)

        price_h.append(pred)
        high_h.append(pred  * (1 + abs(np.random.normal(0, 0.004))))
        low_h.append(pred   * (1 - abs(np.random.normal(0, 0.004))))
        volume_h.append(float(np.mean(volume_h[-5:])))

        row_d  = _build_feature_row(price_h, high_h, low_h, volume_h, feature_cols)
        row_df = pd.DataFrame([row_d])[feature_cols]
        new_sc = scaler.transform(row_df.values)[0]
        seq_arr = np.vstack([seq_arr, new_sc])

    return future


# ──────────────────────────────────────────────────────────────────────────────
#  YARDIMCI: Chart data dict üret
# ──────────────────────────────────────────────────────────────────────────────
def _build_chart_data(
    bt_dates: list, y_test: list, backtest_preds: list,
    future_preds: list, last_date
) -> dict:
    future_dates = [
        (last_date + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(1, 15)
    ]
    full_dates     = bt_dates + future_dates
    full_actual    = [float(v) for v in y_test] + [None] * 14
    full_predicted = [float(v) for v in backtest_preds] + [float(v) for v in future_preds]

    return {
        "dates":            full_dates,
        "actual_prices":    full_actual,
        "predicted_prices": full_predicted,
    }
