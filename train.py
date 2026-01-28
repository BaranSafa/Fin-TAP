import os
import joblib

# Listede görünmesini istediğiniz hisseler
TICKERS_TO_TRAIN = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'AMD', 'CSCO', 'ADBE', 'PYPL', 'NVDA', 'NFLX', 'INTC', 'ORCL', 'IBM', 'CRM', 'QCOM', 'TXN', 'AVGO', 'MU', 'LRCX', 'NOW', 'ZM', 'DOCU', 'SNOW', 'UBER', 'LYFT', 'SPOT', 'TWTR', 'SQ', 'SHOP', 'ETSY']

if __name__ == "__main__":
    # Models klasörünü oluştur
    if not os.path.exists("models"):
        os.makedirs("models")
    
    print("--- Sistem Başlatılıyor ---")
    
    for ticker in TICKERS_TO_TRAIN:
        # Burada veri çekme veya eğitim YAPMIYORUZ.
        # Sadece arayüzde görünebilmesi için boş bir dosya oluşturuyoruz.
        file_path = os.path.join("models", f"{ticker}_scaler.joblib")
        
        # Eğer dosya yoksa oluştur (varsa elleme)
        if not os.path.exists(file_path):
            joblib.dump("dummy_scaler", file_path)
            print(f"'{ticker}' sisteme eklendi.")
        else:
            print(f"'{ticker}' zaten mevcut.")
            
    print("--- Kurulum Tamamlandı ---")
    print("Şimdi 'python run.py' komutunu çalıştırabilirsiniz.")