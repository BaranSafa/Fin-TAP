"""
model_manager.py  —  Fin-TAP Backend
======================================
Bu modülün tek görevi: bir hisse için "AL / SAT / BEKLE" tavsiyesi üretmek.

Nasıl çalışır?
  1. dynamic_trainer'dan 14 günlük fiyat tahmini al (LINEAR model, hız için)
  2. Mevcut fiyatla karşılaştırarak potansiyel kazanç yüzdesini hesapla
  3. RSI, MACD, Bollinger Bantları sinyallerini puanla (score sistemi)
  4. Toplam skora göre AL / SAT / BEKLE kararı ver
  5. Sonucu dashboard'daki "Suggestions" kartlarına gönder
"""
from __future__ import annotations
import traceback

# Paket olarak ve doğrudan çalıştırma için çift import denemesi
try:
    from .dynamic_trainer import train_and_predict_dynamic
    from .data_manager    import get_processed_data
except ImportError:
    from dynamic_trainer import train_and_predict_dynamic
    from data_manager    import get_processed_data


def get_suggestion_metrics(ticker: str) -> dict | None:
    """
    Belirtilen hisse için teknik analiz skorunu hesapla ve öneri döndür.

    Skor sistemi (toplam skor → karar):
      +3 veya daha fazla: STRONG BUY (güçlü al)
      +2 → BUY, 0-1 → HOLD, -2 → CAUTION, -3 ve altı → SELL

    Döndürür: dict (ticker, fiyatlar, RSI, tavsiye, renk kodu)
              None   (veri alınamadıysa)
    """
    try:
        # Adım 1: 14 günlük fiyat tahmini al — LINEAR model, hız öncelikli
        future_preds, _ = train_and_predict_dynamic(
            ticker, "LINEAR", ["RSI", "MACD", "Bollinger", "Volatility"]
        )
        if not future_preds:
            return None

        # Tahmin edilen son fiyat (14. gün)
        future_price = float(future_preds[-1])

        # Anlık kapanış fiyatı ve RSI(14) değerini veri setinden al
        df = get_processed_data(ticker)
        if df is None or df.empty:
            return None

        last_price = float(df["Close"].iloc[-1])
        rsi14      = float(df["rsi_14"].iloc[-1])
        if last_price == 0:
            return None

        # Potansiyel kazanç yüzdesi: (tahmin / anlık - 1) × 100
        gain_pct = ((future_price / last_price) - 1) * 100

        # ── PUANLAMA SİSTEMİ ────────────────────────────────────────────────
        score = 0

        # Fiyat tahminine göre puan:
        if gain_pct > 5:    score += 3   # çok güçlü yükseliş beklentisi
        elif gain_pct > 2:  score += 2
        elif gain_pct > 0:  score += 1
        elif gain_pct < -2: score -= 2   # düşüş beklentisi

        # RSI(14) sinyaline göre puan:
        # RSI < 30 → aşırı satım (oversold) → alım fırsatı
        # RSI > 70 → aşırı alım (overbought) → sat sinyali
        if rsi14 < 30:      score += 3
        elif rsi14 < 45:    score += 1
        elif rsi14 > 70:    score -= 3
        elif rsi14 > 60:    score -= 1

        # MACD sinyaline göre puan:
        # MACD çizgisi sinyal çizgisinin üzerindeyse → yükseliş momentumu
        if "macd" in df.columns and "macd_sig" in df.columns:
            score += 1 if float(df["macd"].iloc[-1]) > float(df["macd_sig"].iloc[-1]) else -1

        # Bollinger Band pozisyonuna göre puan:
        # bb_pct = 0 → alt bant (aşırı satım), bb_pct = 1 → üst bant (aşırı alım)
        if "bb_pct" in df.columns:
            bp = float(df["bb_pct"].iloc[-1])
            if bp < 0.2:   score += 1   # fiyat alt banda yakın → ucuz
            elif bp > 0.8: score -= 1   # fiyat üst banda yakın → pahalı

        # ── SKORA GÖRE TAVSİYE ──────────────────────────────────────────────
        if score >= 4:      rec, risk, color = "STRONG BUY",  "LOW",    "green"
        elif score >= 2:    rec, risk, color = "BUY",         "LOW-MED","green"
        elif score >= 0:    rec, risk, color = "HOLD",        "MEDIUM", "amber"
        elif score >= -2:   rec, risk, color = "CAUTION",     "HIGH",   "amber"
        else:               rec, risk, color = "SELL",        "HIGH",   "red"

        # RSI rengi — aşırı satım yeşil, aşırı alım kırmızı gösterilir
        rsi_color = "text"
        if rsi14 > 70:   rsi_color = "red"
        elif rsi14 < 30: rsi_color = "green"

        return {
            "ticker":             ticker,
            "last_price":         round(last_price, 2),
            "predicted_price":    round(future_price, 2),
            "potential_gain_pct": round(gain_pct, 2),
            "rsi":                round(rsi14, 2),
            "recommendation":     rec,
            "risk_level":         risk,
            "signal_color":       color,
            "rsi_color":          rsi_color,
            "score":              score,
        }
    except Exception as e:
        print(f"[model_manager] {ticker}: {e}")
        traceback.print_exc()
        return None
