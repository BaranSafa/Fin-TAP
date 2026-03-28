"""
app.py  —  Fin-TAP Web Application
Render.com free tier için optimize edilmiş.
"""
from __future__ import annotations

import os, sys, traceback
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from datetime import datetime

from models import db, User, Wallet, Transaction, Prediction

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.append(os.path.dirname(__file__))

try:
    from train import TICKERS_TO_TRAIN
    from backend.dynamic_trainer import train_and_predict_dynamic
    from backend.model_manager   import get_suggestion_metrics
    from backend.data_manager    import get_processed_data
except ImportError as e:
    print(f"[app] Backend import HATASI: {e}")
    traceback.print_exc()
    TICKERS_TO_TRAIN = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]


def get_wallet():
    wallet = Wallet.query.filter_by(user_id=current_user.id).first()
    if not wallet:
        wallet = Wallet(user_id=current_user.id, balance=5)
        db.session.add(wallet)
        db.session.commit()
    return wallet


# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder="frontend/templates",
            static_folder="frontend/static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "fin-tap-dev-secret-change-in-prod")

database_url = os.environ.get("DATABASE_URL", "")
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = database_url or "sqlite:///fintap.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)
db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── Sayfa rotaları ─────────────────────────────────────────────────────────
@app.route("/")
def root():
    if current_user.is_authenticated:
        wallet = get_wallet()
        return render_template("home.html", user=current_user,
                               balance=wallet.balance, stocks=TICKERS_TO_TRAIN)
    return render_template("landing.html")

@app.route("/dashboard")
@login_required
def dashboard():
    wallet = get_wallet()
    return render_template("home.html", user=current_user,
                           balance=wallet.balance, stocks=TICKERS_TO_TRAIN)

@app.route("/predict")
@login_required
def predict():
    wallet = get_wallet()
    ticker = request.args.get("ticker", "AAPL")
    return render_template("predict.html", user=current_user,
                           balance=wallet.balance,
                           trained_stocks=TICKERS_TO_TRAIN,
                           ticker_from_url=ticker)

@app.route("/compare")
@login_required
def compare():
    wallet = get_wallet()
    return render_template("compare.html", user=current_user,
                           balance=wallet.balance, stocks=TICKERS_TO_TRAIN)

@app.route("/all_stocks")
@login_required
def all_stocks():
    wallet = get_wallet()
    return render_template("all_stocks.html", user=current_user,
                           balance=wallet.balance, stocks=TICKERS_TO_TRAIN)

@app.route("/roadmap")
@login_required
def roadmap():
    wallet = get_wallet()
    return render_template("roadmap.html", user=current_user, balance=wallet.balance)

@app.route("/prices")
@login_required
def prices():
    wallet = get_wallet()
    return render_template("prices.html", user=current_user, balance=wallet.balance)

@app.route("/profile")
@login_required
def profile():
    wallet       = get_wallet()
    transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.desc()).all()
    predictions  = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template("profile.html", user=current_user, wallet=wallet,
                           balance=wallet.balance,
                           transactions=transactions, predictions=predictions)


# ── API ────────────────────────────────────────────────────────────────────
@app.route("/api/market_summary")
@login_required
def api_market_summary():
    summary = []
    for ticker in TICKERS_TO_TRAIN[:12]:
        try:
            df = get_processed_data(ticker)
            if df is not None and not df.empty:
                cur  = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                chg  = ((cur - prev) / prev) * 100
                summary.append({"ticker": ticker, "price": round(cur,2),
                                 "change": round(chg,2),
                                 "trend": "up" if chg >= 0 else "down"})
        except Exception as e:
            print(f"[market_summary] {ticker}: {e}")
    return jsonify(summary)


@app.route("/api/history/<ticker>")
@login_required
def api_history(ticker):
    try:
        df = get_processed_data(ticker)
        if df is None or df.empty:
            return jsonify({"error": "Veri yok"}), 404
        recent = df.tail(100)
        return jsonify({
            "dates":  [d.strftime("%Y-%m-%d") for d in recent.index],
            "prices": [round(float(p), 2) for p in recent["Close"].tolist()],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare_stocks")
@login_required
def api_compare():
    t1 = request.args.get("ticker1"); t2 = request.args.get("ticker2")
    if not t1 or not t2:
        return jsonify({"error": "Eksik parametre"}), 400
    m1 = get_suggestion_metrics(t1); m2 = get_suggestion_metrics(t2)
    if not m1 or not m2:
        return jsonify({"error": "Veri alınamadı — lütfen tekrar deneyin"}), 500
    return jsonify({
        t1: {"price": m1["last_price"], "predicted": m1["predicted_price"],
             "gain": m1["potential_gain_pct"], "rsi": m1["rsi"]},
        t2: {"price": m2["last_price"], "predicted": m2["predicted_price"],
             "gain": m2["potential_gain_pct"], "rsi": m2["rsi"]},
    })


@app.route("/api/predict_run", methods=["POST"])
@login_required
def api_predict_run():
    data     = request.json or {}
    ticker   = data.get("ticker", "AAPL")
    model    = data.get("model", "LINEAR")
    features = data.get("features", [])

    wallet = get_wallet()
    if wallet.balance <= 0:
        return jsonify({"error": "Yetersiz bakiye"}), 402

    try:
        preds, chart_data = train_and_predict_dynamic(ticker, model, features)
    except Exception as e:
        print(f"[predict_run] HATA: {e}"); traceback.print_exc()
        return jsonify({"error": f"Model hatası: {str(e)}"}), 500

    if preds is None or len(preds) == 0:
        return jsonify({"error": "Tahmin üretilemedi — lütfen farklı model veya hisse deneyin"}), 500

    wallet.balance -= 1
    db.session.add(Prediction(
        user_id=current_user.id, symbol=ticker, model_type=model,
        predicted_result=f"${round(float(preds[-1]), 2)}"
    ))
    db.session.commit()

    return jsonify({
        "status":     "success",
        "balance":    wallet.balance,
        "prediction": round(float(preds[-1]), 2),
        "chart_data": chart_data,
    })


# ── Auth ───────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("root"))
    if request.method == "POST":
        email    = request.form.get("email")
        name     = request.form.get("name")
        password = request.form.get("password")
        if User.query.filter_by(email=email).first():
            flash("Bu e-posta zaten kayıtlı.", "error")
            return redirect(url_for("register"))
        user = User(email=email, name=name, password=generate_password_hash(password))
        db.session.add(user); db.session.commit()
        db.session.add(Wallet(user_id=user.id, balance=5)); db.session.commit()
        login_user(user)
        return redirect(url_for("root"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("root"))
    if request.method == "POST":
        email    = request.form.get("email")
        password = request.form.get("password")
        user     = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user); return redirect(url_for("root"))
        flash("Hatalı e-posta veya şifre.", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user(); return redirect(url_for("login"))


@app.route("/db-kur")
def db_kur():
    with app.app_context():
        db.create_all()
    return "Veritabanı kuruldu."


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
