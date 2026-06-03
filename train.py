import os
import joblib

# ── Desteklenen hisse senetleri (Yahoo Finance sembol formatı) ───────────────
STOCKS = [
    'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'AMD', 'CSCO', 'ADBE',
    'PYPL', 'NVDA', 'NFLX', 'INTC', 'ORCL', 'IBM', 'CRM', 'QCOM',
    'TXN', 'AVGO', 'MU', 'LRCX', 'NOW', 'ZM', 'DOCU', 'SNOW',
    'UBER', 'LYFT', 'SPOT', 'SQ', 'SHOP', 'ETSY',
]

# ── Desteklenen kripto paralar — yfinance BTC-USD formatını kullanır ─────────
CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD',
    'ADA-USD', 'XRP-USD', 'DOGE-USD', 'AVAX-USD',
]

# app.py bu listeyi import eder → hangi sembollerin geçerli olduğunu bilir
TICKERS_TO_TRAIN = STOCKS + CRYPTO

if __name__ == "__main__":
    # models/ klasörü yoksa oluştur
    if not os.path.exists("models"):
        os.makedirs("models")

    print("--- Fin-TAP Sistem Başlatılıyor ---")

    for ticker in TICKERS_TO_TRAIN:
        # Tire işaretini alt çizgiye çevir → dosya adı sorunsuz (örn. BTC-USD → BTC_USD)
        file_path = os.path.join("models", f"{ticker.replace('-','_')}_scaler.joblib")
        if not os.path.exists(file_path):
            # Gerçek scaler değil, sadece "bu sembol tanınıyor" işareti olarak kaydedilir
            joblib.dump("dummy_scaler", file_path)
            print(f"  ✓ '{ticker}' sisteme eklendi.")
        else:
            print(f"  · '{ticker}' zaten mevcut.")

    print(f"\n--- Kurulum Tamamlandı ({len(TICKERS_TO_TRAIN)} araç) ---")
    print("Şimdi 'python run.py' komutunu çalıştırabilirsiniz.")
