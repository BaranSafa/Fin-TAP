import os
import joblib

# Hisse senetleri
STOCKS = [
    'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'AMD', 'CSCO', 'ADBE',
    'PYPL', 'NVDA', 'NFLX', 'INTC', 'ORCL', 'IBM', 'CRM', 'QCOM',
    'TXN', 'AVGO', 'MU', 'LRCX', 'NOW', 'ZM', 'DOCU', 'SNOW',
    'UBER', 'LYFT', 'SPOT', 'SQ', 'SHOP', 'ETSY',
]

# Kripto paralar — yfinance BTC-USD formatını destekler
CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD',
    'ADA-USD', 'XRP-USD', 'DOGE-USD', 'AVAX-USD',
]

TICKERS_TO_TRAIN = STOCKS + CRYPTO

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    print("--- Fin-TAP Sistem Başlatılıyor ---")

    for ticker in TICKERS_TO_TRAIN:
        file_path = os.path.join("models", f"{ticker.replace('-','_')}_scaler.joblib")
        if not os.path.exists(file_path):
            joblib.dump("dummy_scaler", file_path)
            print(f"  ✓ '{ticker}' sisteme eklendi.")
        else:
            print(f"  · '{ticker}' zaten mevcut.")

    print(f"\n--- Kurulum Tamamlandı ({len(TICKERS_TO_TRAIN)} araç) ---")
    print("Şimdi 'python run.py' komutunu çalıştırabilirsiniz.")
