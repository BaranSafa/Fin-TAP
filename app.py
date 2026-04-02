"""
app.py  —  Fin-TAP  (Render production-grade)

V0.7 Güncellemeleri:
  1. Flask-Limiter: Rate limiting eklendi (DDoS koruması)
  2. Flask-WTF: CSRF koruması form route'larına eklendi
  3. Güvenli SECRET_KEY: Env var yoksa rastgele üretir
  4. Email enumeration düzeltildi (generic hata mesajı)
  5. /db-test ve /db-kur ADMIN_SECRET ile korundu
  6. Ticker ve feature group input validation eklendi
  7. Şifre güvenlik kontrolü backend'e taşındı (min 8 karakter)
  8. Watchlist CRUD endpoint'leri eklendi
  9. Tahmin doğruluk takibi endpoint'i eklendi
  10. Token bakiyesi prediction response'da anlık döndürülüyor
"""
from __future__ import annotations

import os, sys, traceback, secrets
from sqlalchemy import inspect as db_inspect
from flask import (Flask, render_template, redirect, url_for,
                   flash, request, jsonify, session)
from flask_login import (LoginManager, login_user, login_required,
                         logout_user, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect, generate_csrf

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, BASE_DIR)

from models import db, User, Wallet, Transaction, Prediction, Watchlist

# ── Backend ────────────────────────────────────────────────────────────────
try:
    from train import TICKERS_TO_TRAIN
except ImportError:
    TICKERS_TO_TRAIN = ["AAPL","GOOG","MSFT","AMZN","TSLA","NVDA","AMD","NFLX"]

try:
    from backend.dynamic_trainer import train_and_predict_dynamic
    from backend.model_manager   import get_suggestion_metrics
    from backend.data_manager    import get_processed_data, FEATURE_GROUPS
    print("[app] Backend modülleri yüklendi OK")
except ImportError as e:
    print(f"[app] Backend import hatası: {e}")
    traceback.print_exc()
    def train_and_predict_dynamic(*a, **kw): return None, None
    def get_suggestion_metrics(*a, **kw):    return None
    def get_processed_data(*a, **kw):        return None
    FEATURE_GROUPS = {}

# Geçerli ticker ve feature group listeleri (input validation için)
VALID_TICKERS       = set(TICKERS_TO_TRAIN)
VALID_FEATURE_GROUPS = set(FEATURE_GROUPS.keys()) if FEATURE_GROUPS else {
    "Returns","RSI","MACD","Bollinger","SMA","EMA","Volatility",
    "ATR","Stoch","Williams","CCI","ADX","Momentum","Volume","Pattern",
    "Distance","Trend"
}

# ── Flask ──────────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "frontend", "templates"),
            static_folder=os.path.join(BASE_DIR, "frontend", "static"))

# SECRET_KEY: env var yoksa güvenli rastgele üret (her yeniden başlatmada session geçersiz olur)
_secret = os.environ.get("SECRET_KEY")
if not _secret:
    _secret = secrets.token_hex(32)
    print("[app] UYARI: SECRET_KEY env var eksik! Render > Environment'a ekle. "
          "Geçici rastgele key üretildi (restart'ta session'lar sıfırlanır).")
app.config["SECRET_KEY"] = _secret

# ── DB ──────────────────────────────────────────────────────────────────────
_db_url = os.environ.get("DATABASE_URL", "")
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)

if _db_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = _db_url
    print("[app] PostgreSQL kullanılıyor")
else:
    _inst = os.path.join(BASE_DIR, "instance")
    os.makedirs(_inst, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{_inst}/fintap.db"
    print("[app] SQLite kullanılıyor (Render'da her deploy sıfırlanır!)")

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle":  280,
}
app.config["WTF_CSRF_TIME_LIMIT"] = 3600   # CSRF token 1 saat geçerli

# ── CORS ──────────────────────────────────────────────────────────────────
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
if _allowed_origins == "*":
    print("[app] UYARI: CORS tüm origin'lere açık. Prod'da ALLOWED_ORIGINS env var'ını ayarla.")
CORS(app, origins=_allowed_origins)

# ── Extensions ────────────────────────────────────────────────────────────
db.init_app(app)
csrf = CSRFProtect(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["300 per day", "60 per hour"],
    storage_uri="memory://",
)

login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))

# DB tablolarını oluştur
with app.app_context():
    try:
        db.create_all()
        print("[app] DB tabloları hazır OK")
    except Exception as e:
        print(f"[app] db.create_all hatası: {e}")

    # V0.7 migrasyon: prediction tablosuna accuracy_pct kolonu ekle (yoksa)
    try:
        db.session.execute(db.text("ALTER TABLE prediction ADD COLUMN accuracy_pct FLOAT"))
        db.session.commit()
        print("[app] Migrasyon: prediction.accuracy_pct kolonu eklendi OK")
    except Exception:
        db.session.rollback()
        # Kolon zaten mevcut, sorun yok


# ── CSRF token'ı her response'a ekle (JS fetch için) ──────────────────────
@app.after_request
def inject_csrf_token(response):
    response.set_cookie("csrf_token", generate_csrf())
    return response


# ── Yardımcı ───────────────────────────────────────────────────────────────
def get_wallet():
    w = Wallet.query.filter_by(user_id=current_user.id).first()
    if not w:
        w = Wallet(user_id=current_user.id, balance=5)
        db.session.add(w); db.session.commit()
    return w


def _admin_check() -> bool:
    """Admin secret kontrolü — /db-test ve /db-kur için."""
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    if not admin_secret:
        return True  # Env var ayarlanmamışsa geliştirme modunda izin ver
    return request.args.get("secret", "") == admin_secret


# ══════════════════════════════════════════════════════════════════════════
#  SAYFALAR
# ══════════════════════════════════════════════════════════════════════════

@app.route("/ping")
def ping():
    """UptimeRobot bu endpoint'i her 5 dakika ping atar → Render uyumaz."""
    return "OK", 200


# ── Cache & veri güncelleme endpoint'leri ─────────────────────────────────────

@app.route("/api/cache/status")
@login_required
@limiter.limit("30 per minute")
def api_cache_status():
    try:
        from backend.data_manager import cache_status
        status = cache_status()
        return jsonify({
            "cache": status,
            "total_cached": len(status),
            "market_open": _market_open_check(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/clear")
@login_required
@limiter.limit("10 per hour")
def api_cache_clear():
    ticker = request.args.get("ticker")
    try:
        from backend.data_manager import cache_clear
        cache_clear(ticker)
        return jsonify({"status": "ok", "cleared": ticker or "all"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh")
@limiter.limit("5 per hour")
def api_refresh():
    """
    Cron job bu endpoint'i çağırır → tüm tickerları günceller.
    CRON_SECRET env var ayarlanmışsa kimlik doğrulama zorunlu.
    """
    secret   = request.args.get("secret", "")
    expected = os.environ.get("CRON_SECRET", "")

    if expected and secret != expected:
        return jsonify({"error": "unauthorized"}), 401

    results = {"refreshed": [], "failed": [], "skipped": []}

    try:
        from backend.data_manager import get_processed_data, cache_clear
        cache_clear()

        for ticker in TICKERS_TO_TRAIN[:10]:
            try:
                df = get_processed_data(ticker, force_refresh=True)
                if df is not None:
                    results["refreshed"].append(ticker)
                    print(f"[refresh] {ticker}: OK, son: {df.index[-1].date()}")
                else:
                    results["failed"].append(ticker)
            except Exception as e:
                print(f"[refresh] {ticker}: HATA: {e}")
                results["failed"].append(ticker)

        results["skipped"] = TICKERS_TO_TRAIN[10:]
        print(f"[refresh] Tamamlandı: {len(results['refreshed'])} OK, "
              f"{len(results['failed'])} başarısız")

    except Exception as e:
        print(f"[refresh] HATA: {e}"); traceback.print_exc()
        results["error"] = str(e)

    return jsonify(results)


def _market_open_check():
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime as dt
        now_et = dt.now(tz=ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        t = now_et.hour * 60 + now_et.minute
        return 570 <= t < 960   # 09:30–16:00 ET
    except Exception:
        from datetime import datetime
        now = datetime.utcnow()
        if now.weekday() >= 5:
            return False
        return 14 <= now.hour < 21


@app.route("/db-test")
def db_test():
    """DB bağlantısını test et — ADMIN_SECRET ile korumalı."""
    if not _admin_check():
        return jsonify({"error": "Yetkisiz erişim"}), 403

    info = {
        "db_url_set":    bool(os.environ.get("DATABASE_URL")),
        "db_url_prefix": (os.environ.get("DATABASE_URL") or "")[:30] + "...",
        "sqlalchemy_uri": app.config["SQLALCHEMY_DATABASE_URI"][:40] + "...",
    }
    try:
        db.session.execute(db.text("SELECT 1"))
        info["connection"] = "OK ✓"
        info["tables"]     = db_inspect(db.engine).get_table_names()
    except Exception as e:
        info["connection"] = f"HATA: {str(e)[:200]}"
    return jsonify(info), 200


@app.route("/db-kur")
def db_kur():
    """
    Tabloları yarat veya güncelle — ADMIN_SECRET ile korumalı.
    İlk deploy veya şema değişikliği sonrası bir kez ziyaret et.
    """
    if not _admin_check():
        return jsonify({"error": "Yetkisiz erişim"}), 403

    try:
        with app.app_context():
            db.create_all()

            _db_url = app.config.get("SQLALCHEMY_DATABASE_URI", "")
            if "postgresql" in _db_url:
                try:
                    db.session.execute(db.text(
                        "ALTER TABLE \"user\" "
                        "ALTER COLUMN password TYPE VARCHAR(512)"
                    ))
                    db.session.commit()
                    print("[db-kur] password kolonu 512'ye genişletildi")
                except Exception as alter_e:
                    db.session.rollback()
                    print(f"[db-kur] ALTER zaten yapılmış veya hata: {alter_e}")

            # V0.7 migrasyon: accuracy_pct kolonu (PostgreSQL ve SQLite için)
            try:
                db.session.execute(db.text(
                    "ALTER TABLE prediction ADD COLUMN accuracy_pct FLOAT"
                ))
                db.session.commit()
                print("[db-kur] prediction.accuracy_pct kolonu eklendi")
            except Exception as alter_e:
                db.session.rollback()
                print(f"[db-kur] accuracy_pct zaten mevcut veya hata: {alter_e}")

        return "Veritabanı kuruldu ve güncellendi ✓", 200
    except Exception as e:
        return f"Hata: {e}", 500


@app.route("/")
def root():
    if current_user.is_authenticated:
        w = get_wallet()
        return render_template("home.html", user=current_user,
                               balance=w.balance, stocks=TICKERS_TO_TRAIN)
    return render_template("landing.html")


@app.route("/dashboard")
@login_required
def dashboard():
    w = get_wallet()
    return render_template("home.html", user=current_user,
                           balance=w.balance, stocks=TICKERS_TO_TRAIN)


@app.route("/predict")
@login_required
def predict():
    w = get_wallet()
    return render_template("predict.html", user=current_user,
                           balance=w.balance,
                           trained_stocks=TICKERS_TO_TRAIN,
                           ticker_from_url=request.args.get("ticker", "AAPL"))


@app.route("/compare")
@login_required
def compare():
    w = get_wallet()
    return render_template("compare.html", user=current_user,
                           balance=w.balance, stocks=TICKERS_TO_TRAIN)


@app.route("/all_stocks")
@login_required
def all_stocks():
    w = get_wallet()
    return render_template("all_stocks.html", user=current_user,
                           balance=w.balance, stocks=TICKERS_TO_TRAIN)


@app.route("/roadmap")
@login_required
def roadmap():
    w = get_wallet()
    return render_template("roadmap.html", user=current_user, balance=w.balance)


@app.route("/prices")
@login_required
def prices():
    w = get_wallet()
    return render_template("prices.html", user=current_user, balance=w.balance)


@app.route("/profile")
@login_required
def profile():
    w   = get_wallet()
    txs = Transaction.query.filter_by(user_id=current_user.id)\
                     .order_by(Transaction.date.desc()).limit(50).all()
    prs = Prediction.query.filter_by(user_id=current_user.id)\
                    .order_by(Prediction.created_at.desc()).limit(100).all()
    return render_template("profile.html", user=current_user, wallet=w,
                           balance=w.balance, transactions=txs, predictions=prs)


# ══════════════════════════════════════════════════════════════════════════
#  API
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/market_summary")
@login_required
@limiter.limit("30 per minute")
@csrf.exempt
def api_market_summary():
    result = []
    for t in TICKERS_TO_TRAIN[:12]:
        try:
            df = get_processed_data(t)
            if df is not None and not df.empty:
                cur  = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                chg  = ((cur - prev) / prev) * 100
                result.append({"ticker": t, "price": round(cur, 2),
                                "change": round(chg, 2),
                                "trend": "up" if chg >= 0 else "down"})
        except Exception as e:
            print(f"[market_summary] {t}: {e}")
    return jsonify(result)


@app.route("/api/history/<ticker>")
@login_required
@limiter.limit("60 per minute")
@csrf.exempt
def api_history(ticker):
    # Ticker whitelist kontrolü
    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Geçersiz ticker"}), 400
    try:
        df = get_processed_data(ticker)
        if df is None or df.empty:
            return jsonify({"error": "Veri alınamadı"}), 404
        r = df.tail(100)
        return jsonify({
            "dates":  [d.strftime("%Y-%m-%d") for d in r.index],
            "prices": [round(float(p), 2) for p in r["Close"]],
        })
    except Exception as e:
        print(f"[api_history] {ticker}: {e}"); traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare_stocks")
@login_required
@limiter.limit("20 per minute")
@csrf.exempt
def api_compare():
    t1 = request.args.get("ticker1", "").upper()
    t2 = request.args.get("ticker2", "").upper()

    if not t1 or not t2:
        return jsonify({"error": "Eksik parametre"}), 400
    if t1 not in VALID_TICKERS or t2 not in VALID_TICKERS:
        return jsonify({"error": "Geçersiz ticker"}), 400

    try:
        m1 = get_suggestion_metrics(t1)
        m2 = get_suggestion_metrics(t2)
    except Exception as e:
        print(f"[api_compare]: {e}"); traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    if not m1 or not m2:
        return jsonify({"error": "Tahmin üretilemedi — LINEAR modeli deneyin"}), 500
    return jsonify({
        t1: {"price": m1["last_price"], "predicted": m1["predicted_price"],
             "gain":  m1["potential_gain_pct"], "rsi": m1["rsi"]},
        t2: {"price": m2["last_price"], "predicted": m2["predicted_price"],
             "gain":  m2["potential_gain_pct"], "rsi": m2["rsi"]},
    })


VALID_MODELS = {"LINEAR","RANDOM_FOREST","EXTRA_TREES","GRADIENT_BOOST",
                "XGBOOST","LIGHTGBM","LSTM"}


@app.route("/api/predict_run", methods=["POST"])
@login_required
@limiter.limit("10 per hour")
@csrf.exempt
def api_predict_run():
    payload  = request.get_json(silent=True) or {}
    ticker   = str(payload.get("ticker", "AAPL")).upper().strip()
    model    = str(payload.get("model",  "LINEAR")).upper().strip()
    features = payload.get("features", [])

    # Input validation
    if ticker not in VALID_TICKERS:
        return jsonify({"error": f"Geçersiz ticker: {ticker}"}), 400
    if model not in VALID_MODELS:
        return jsonify({"error": f"Geçersiz model: {model}"}), 400
    if not isinstance(features, list):
        return jsonify({"error": "features bir liste olmalı"}), 400
    invalid_feats = [f for f in features if f not in VALID_FEATURE_GROUPS]
    if invalid_feats:
        return jsonify({"error": f"Geçersiz feature group(lar): {invalid_feats}"}), 400

    w = get_wallet()
    if w.balance <= 0:
        return jsonify({"error": "Yetersiz bakiye"}), 402

    print(f"[predict_run] {ticker} | {model} | {len(features)} feature")

    try:
        preds, chart_data = train_and_predict_dynamic(ticker, model, features)
    except Exception as e:
        print(f"[predict_run] EXCEPTION: {e}"); traceback.print_exc()
        return jsonify({"error": f"Model istisnası: {str(e)[:300]}"}), 500

    if not preds:
        return jsonify({
            "error": (
                "Tahmin başarısız. "
                "Öneri: LINEAR modeli ve az feature seçin, "
                "veya farklı bir hisse deneyin."
            )
        }), 500

    w.balance -= 1
    w.last_updated = __import__("datetime").datetime.utcnow()
    try:
        db.session.add(Prediction(
            user_id=current_user.id, symbol=ticker, model_type=model,
            predicted_result=f"${round(float(preds[-1]), 2)}"
        ))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"[predict_run] DB kayıt hatası: {e}")

    return jsonify({
        "status":     "success",
        "balance":    w.balance,
        "prediction": round(float(preds[-1]), 2),
        "chart_data": chart_data,
    })


# ── Watchlist API ──────────────────────────────────────────────────────────

@app.route("/api/watchlist", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
@csrf.exempt
def api_watchlist_get():
    """Kullanıcının takip listesini döndür."""
    items = Watchlist.query.filter_by(user_id=current_user.id)\
                    .order_by(Watchlist.added_at.desc()).all()
    result = []
    for item in items:
        try:
            df = get_processed_data(item.symbol)
            price = change = None
            if df is not None and not df.empty:
                cur   = float(df["Close"].iloc[-1])
                prev  = float(df["Close"].iloc[-2])
                price  = round(cur, 2)
                change = round(((cur - prev) / prev) * 100, 2)
        except Exception:
            price = change = None
        result.append({
            "symbol":   item.symbol,
            "added_at": item.added_at.strftime("%Y-%m-%d"),
            "price":    price,
            "change":   change,
        })
    return jsonify(result)


@app.route("/api/watchlist", methods=["POST"])
@login_required
@limiter.limit("30 per hour")
@csrf.exempt
def api_watchlist_add():
    """Takip listesine hisse ekle."""
    payload = request.get_json(silent=True) or {}
    symbol  = str(payload.get("symbol", "")).upper().strip()

    if not symbol:
        return jsonify({"error": "symbol gerekli"}), 400
    if symbol not in VALID_TICKERS:
        return jsonify({"error": "Geçersiz ticker"}), 400

    existing = Watchlist.query.filter_by(
        user_id=current_user.id, symbol=symbol
    ).first()
    if existing:
        return jsonify({"status": "already_exists", "symbol": symbol})

    item = Watchlist(user_id=current_user.id, symbol=symbol)
    db.session.add(item)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "added", "symbol": symbol})


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
@login_required
@limiter.limit("30 per hour")
@csrf.exempt
def api_watchlist_remove(symbol):
    """Takip listesinden hisse çıkar."""
    symbol = symbol.upper().strip()
    item   = Watchlist.query.filter_by(
        user_id=current_user.id, symbol=symbol
    ).first()
    if not item:
        return jsonify({"error": "Bulunamadı"}), 404
    db.session.delete(item)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "removed", "symbol": symbol})


# ── Prediction Accuracy API ────────────────────────────────────────────────

@app.route("/api/accuracy/update")
@login_required
@limiter.limit("5 per hour")
@csrf.exempt
def api_accuracy_update():
    """
    Geçmiş tahminlerin doğruluğunu günceller.
    Tahmin tarihinden 14 gün sonra gerçek kapanış fiyatını çekip karşılaştırır.
    """
    from datetime import datetime, timedelta
    updated = 0
    errors  = 0

    preds = Prediction.query.filter_by(
        user_id=current_user.id, accuracy_pct=None
    ).filter(
        Prediction.created_at < datetime.utcnow() - timedelta(days=14)
    ).limit(20).all()

    for pred in preds:
        try:
            df = get_processed_data(pred.symbol)
            if df is None or df.empty:
                continue

            target_date = pred.created_at + timedelta(days=14)
            future_rows = df[df.index >= target_date]
            if future_rows.empty:
                continue

            actual_price = float(future_rows["Close"].iloc[0])
            if not pred.predicted_result:
                continue

            predicted_price = float(pred.predicted_result.replace("$", ""))
            accuracy = 100 - abs((actual_price - predicted_price) / actual_price * 100)
            pred.accuracy_pct = round(max(accuracy, 0), 2)
            updated += 1
        except Exception as e:
            print(f"[accuracy_update] {pred.id}: {e}")
            errors += 1

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

    return jsonify({"updated": updated, "errors": errors})


# ══════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════

def _password_strong(pw: str) -> tuple[bool, str]:
    """Şifre güvenlik kontrolü. (bool, hata_mesajı)"""
    if len(pw) < 8:
        return False, "Şifre en az 8 karakter olmalı."
    has_letter = any(c.isalpha() for c in pw)
    has_digit  = any(c.isdigit() for c in pw)
    if not has_letter or not has_digit:
        return False, "Şifre en az 1 harf ve 1 rakam içermeli."
    return True, ""


@app.route("/register", methods=["GET", "POST"])
@limiter.limit("10 per hour")
def register():
    if current_user.is_authenticated:
        return redirect(url_for("root"))
    if request.method == "POST":
        email    = (request.form.get("email")    or "").strip().lower()
        name     = (request.form.get("name")     or "").strip()
        password =  request.form.get("password") or ""

        if not email or not name or not password:
            flash("Tüm alanları doldurun.", "error")
            return redirect(url_for("register"))

        ok, msg = _password_strong(password)
        if not ok:
            flash(msg, "error")
            return redirect(url_for("register"))

        # Email enumeration'ı önle: aynı generic hata mesajını kullan
        if User.query.filter_by(email=email).first():
            flash("Kayıt tamamlanamadı. Lütfen farklı bir e-posta deneyin.", "error")
            return redirect(url_for("register"))

        try:
            hashed_pw = generate_password_hash(password)
            user   = User(email=email, name=name, password=hashed_pw)
            db.session.add(user)
            db.session.flush()
            wallet = Wallet(user_id=user.id, balance=5)
            db.session.add(wallet)
            db.session.commit()
            login_user(user)
            print(f"[register] Yeni kullanıcı: {email}")
            return redirect(url_for("root"))
        except Exception as e:
            db.session.rollback()
            print(f"[register] HATA: {type(e).__name__}: {e}")
            traceback.print_exc()
            flash("Kayıt sırasında bir hata oluştu. Lütfen tekrar deneyin.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("20 per hour")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("root"))
    if request.method == "POST":
        email    = (request.form.get("email")    or "").strip().lower()
        password =  request.form.get("password") or ""
        user     = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(request.args.get("next") or url_for("root"))
        flash("Hatalı e-posta veya şifre.", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ── Hata handler'ları ──────────────────────────────────────────────────────
@app.errorhandler(404)
def e404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint bulunamadı"}), 404
    return redirect(url_for("root"))

@app.errorhandler(429)
def rate_limit_exceeded(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Çok fazla istek. Lütfen bekleyin."}), 429
    flash("Çok fazla istek gönderildi. Lütfen biraz bekleyin.", "error")
    return redirect(url_for("root"))

@app.errorhandler(500)
def e500(e):
    print(f"[500 handler] {e}"); traceback.print_exc()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Sunucu hatası — logları kontrol et"}), 500
    return "<h3>Sunucu hatası. Birkaç saniye bekleyip tekrar deneyin.</h3>", 500

@app.errorhandler(Exception)
def unhandled(e):
    print(f"[unhandled exception] {e}"); traceback.print_exc()
    if request.path.startswith("/api/"):
        return jsonify({"error": str(e)[:200]}), 500
    return "<h3>Beklenmeyen hata. Lütfen tekrar deneyin.</h3>", 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
