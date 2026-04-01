"""
app.py  —  Fin-TAP  (Render production-grade)

Düzeltilen 500 hata nedenleri:
  1. db.create_all() uygulama başlarken app context içinde çalışıyor
  2. SQLAlchemy 2.0 uyumu: User.query.get() → db.session.get()
  3. SQLite instance/ klasörü otomatik yaratılıyor
  4. SECRET_KEY eksikse uyarı verir, yine de çalışır
  5. pool_pre_ping + pool_recycle → DB bağlantısı kopunca yeniden bağlanır
  6. /ping endpoint → UptimeRobot (uyku engelleme)
  7. 404 ve 500 hata handler'ları eklendi
  8. Tüm exception'lar Render loglarında görünür
"""
from __future__ import annotations

import os, sys, traceback
from flask import (Flask, render_template, redirect, url_for,
                   flash, request, jsonify)
from flask_login import (LoginManager, login_user, login_required,
                         logout_user, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, BASE_DIR)

from models import db, User, Wallet, Transaction, Prediction

# ── Backend ────────────────────────────────────────────────────────────────
try:
    from train import TICKERS_TO_TRAIN
except ImportError:
    TICKERS_TO_TRAIN = ["AAPL","GOOG","MSFT","AMZN","TSLA","NVDA","AMD","NFLX"]

try:
    from backend.dynamic_trainer import train_and_predict_dynamic
    from backend.model_manager   import get_suggestion_metrics
    from backend.data_manager    import get_processed_data
    print("[app] Backend modülleri yüklendi OK")
except ImportError as e:
    print(f"[app] Backend import hatası: {e}")
    traceback.print_exc()
    def train_and_predict_dynamic(*a, **kw): return None, None
    def get_suggestion_metrics(*a, **kw):    return None
    def get_processed_data(*a, **kw):        return None

# ── Flask ──────────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "frontend", "templates"),
            static_folder=os.path.join(BASE_DIR, "frontend", "static"))

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "fintap-fallback-key-set-env-var")

if not os.environ.get("SECRET_KEY"):
    print("[app] UYARI: SECRET_KEY env var eksik! Render > Environment'a ekle.")

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

CORS(app)
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))   # SQLAlchemy 2.0 uyumlu

# DB tablolarını oluştur
with app.app_context():
    try:
        db.create_all()
        print("[app] DB tabloları hazır OK")
    except Exception as e:
        print(f"[app] db.create_all hatası: {e}")


# ── Yardımcı ───────────────────────────────────────────────────────────────
def get_wallet():
    w = Wallet.query.filter_by(user_id=current_user.id).first()
    if not w:
        w = Wallet(user_id=current_user.id, balance=5)
        db.session.add(w); db.session.commit()
    return w


# ══════════════════════════════════════════════════════════════════════════
#  SAYFALAR
# ══════════════════════════════════════════════════════════════════════════

@app.route("/ping")
def ping():
    """UptimeRobot bu endpoint'i her 5 dakika ping atar → Render uyumaz."""
    return "OK", 200


@app.route("/db-kur")
def db_kur():
    """
    Tabloları yarat veya güncelle.
    İlk deploy veya şema değişikliği sonrası bir kez ziyaret et:
    https://fin-tap.onrender.com/db-kur
    """
    try:
        with app.app_context():
            db.create_all()

            # password kolonu 150 → 512 migration (PostgreSQL için)
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
                     .order_by(Transaction.date.desc()).all()
    prs = Prediction.query.filter_by(user_id=current_user.id)\
                    .order_by(Prediction.created_at.desc()).all()
    return render_template("profile.html", user=current_user, wallet=w,
                           balance=w.balance, transactions=txs, predictions=prs)


# ══════════════════════════════════════════════════════════════════════════
#  API
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/market_summary")
@login_required
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
def api_history(ticker):
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
def api_compare():
    t1 = request.args.get("ticker1")
    t2 = request.args.get("ticker2")
    if not t1 or not t2:
        return jsonify({"error": "Eksik parametre"}), 400
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


@app.route("/api/predict_run", methods=["POST"])
@login_required
def api_predict_run():
    payload  = request.get_json(silent=True) or {}
    ticker   = payload.get("ticker",   "AAPL")
    model    = payload.get("model",    "LINEAR")
    features = payload.get("features", [])

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


# ══════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════

@app.route("/register", methods=["GET", "POST"])
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
        if User.query.filter_by(email=email).first():
            flash("Bu e-posta zaten kayıtlı.", "error")
            return redirect(url_for("register"))
        try:
            hashed_pw = generate_password_hash(password)
            print(f"[register] hash uzunluğu: {len(hashed_pw)}")  # debug
            user = User(email=email, name=name, password=hashed_pw)
            db.session.add(user)
            db.session.flush()   # user.id'yi al, commit etme henüz
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
            flash(f"Kayıt hatası: {str(e)[:100]}", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
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
