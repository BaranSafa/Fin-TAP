import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from .data_manager import get_processed_data

# GÜVENLİ İMPORTLAR
try: import xgboost as xgb
except: xgb = None
try: import lightgbm as lgb
except: lgb = None
try: 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Input
    HAS_TF = True
except: HAS_TF = False

FEATURE_GROUPS = {
    'OHLCV': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'RSI': ['RSI_14'], 
    'MACD': ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'],
    'SMA': ['SMA_14', 'SMA_50'], 
    'EMA': ['EMA_14'],
    'Bollinger': ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'],
    'ATR': ['ATRr_14'], 
    'Stoch': ['STOCHk_14_3_3', 'STOCHd_14_3_3'],
    'Returns': ['returns'], 
    'Lag': ['lag_1', 'lag_2'], 
    'Momentum': ['MOM_10'],
    'Volatility': ['volatility'], 
    'OBV': ['OBV'], 
    'ADX': ['ADX_14', 'DMP_14', 'DMN_14', 'ADXR_14_2']
}

def train_and_predict_dynamic(ticker, model_type, selected_feature_groups):
    # 1. Veriyi Al (Son günler DAHİL)
    df = get_processed_data(ticker)
    if df is None or df.empty: return None, None

    # 2. Özellikleri Seç
    features = FEATURE_GROUPS['OHLCV'].copy()
    for g in selected_feature_groups:
        if g in FEATURE_GROUPS:
            features.extend([c for c in FEATURE_GROUPS[g] if c in df.columns])
    features = list(set(features))
    
    # 3. Veriyi Hazırla
    # X_full: Tüm veriler (Bugün dahil)
    X_full = df[features].values
    
    # Scaler'ı tüm veri üzerinde eğit
    scaler = MinMaxScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    
    # --- KRİTİK NOKTA: Gelecek Tahmini İçin Girdi ---
    # En son satır (Bugünün kapanışı ve indikatörleri)
    last_data_point = X_full_scaled[-1:].copy()
    
    # --- EĞİTİM VERİSİ HAZIRLAMA ---
    # Hedef: Yarının fiyatı (Close sütununu 1 gün kaydır)
    # shift(-1) yapınca SON SATIRIN hedefi NaN olur (çünkü yarın yok).
    y_full = df['Close'].shift(-1).values
    
    # NaN olan son satırı eğitimden çıkarıyoruz
    # (Ama last_data_point olarak yukarıda sakladık, onu kaybetmedik!)
    X_train_data = X_full_scaled[:-1] # Son satır hariç hepsi
    y_train_data = y_full[:-1]        # Son satır hariç hepsi
    
    # Train/Test Split (Backtest için)
    split_idx = int(len(X_train_data) * 0.95)
    X_train, y_train = X_train_data[:split_idx], y_train_data[:split_idx]
    X_test, y_test = X_train_data[split_idx:], y_train_data[split_idx:]
    
    # Test Tarihleri (Grafik için)
    # df.index[:-1] çünkü son günü eğitimden çıkardık
    test_dates = [d.strftime('%Y-%m-%d') for d in df.index[:-1][split_idx:]]

    try:
        model = None
        if model_type == 'LINEAR':
            model = LinearRegression().fit(X_train, y_train)
        elif model_type == 'RANDOM_FOREST':
            model = RandomForestRegressor(n_estimators=50).fit(X_train, y_train)
        elif model_type == 'XGBOOST':
            if not xgb: raise ImportError("XGBoost eksik")
            model = xgb.XGBRegressor(n_estimators=200).fit(X_train, y_train)
        elif model_type == 'LIGHTGBM':
            if not lgb: raise ImportError("LightGBM eksik")
            model = lgb.LGBMRegressor(n_estimators=200, verbose=-1).fit(X_train, y_train)
        elif model_type == 'LSTM':
            if not HAS_TF: raise ImportError("Tensorflow eksik")
            X_tr_r = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            model = Sequential([Input(shape=(1, X_train.shape[1])), LSTM(50), Dense(1)])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_tr_r, y_train, epochs=5, verbose=0)
        else: return None, None

        # --- 14 GÜNLÜK GELECEK TAHMİNİ (RECURSIVE) ---
        future_predictions = []
        current_input = last_data_point # BUGÜNÜN verisinden başla
        
        for _ in range(14):
            if model_type == 'LSTM':
                curr_reshaped = current_input.reshape((1, 1, current_input.shape[1]))
                pred = model.predict(curr_reshaped, verbose=0).flatten()[0]
            else:
                pred = model.predict(current_input)[0]
            
            future_predictions.append(float(pred))
            
            # Bir sonraki gün için inputu güncelle
            # (Basit simülasyon: Inputu hafif değiştirerek döngüye sok)
            variation = np.random.normal(0, 0.002) 
            current_input = current_input * (1 + variation)

        # Backtest Sonuçları (Turuncu Çizgi)
        if model_type == 'LSTM':
             X_test_r = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
             backtest_preds = model.predict(X_test_r, verbose=0).flatten()
        else:
             backtest_preds = model.predict(X_test)

        return future_predictions, {
            "dates": test_dates,
            "actual_prices": [float(v) for v in y_test],
            "predicted_prices": [float(v) for v in backtest_preds]
        }

    except Exception as e:
        print(f"Eğitim hatası: {e}")
        return None, None