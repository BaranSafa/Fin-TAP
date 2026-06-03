from __future__ import annotations

import os, sys, traceback, secrets, json, hashlib, math
from datetime import datetime as _dt
from urllib.parse import urlparse, urljoin
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from sqlalchemy import inspect as db_inspect, func as _func
from flask import (Flask, render_template, redirect, url_for,
                   flash, request, jsonify, session)
from flask_login import (LoginManager, login_user, login_required,
                         logout_user, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_mail import Mail, Message

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, BASE_DIR)

from models import (db, User, Wallet, Transaction, Prediction, Watchlist,
                    PriceAlert, PaperPortfolio, PaperPosition, PaperTrade,
                    PAPER_STARTING_CASH, ApiKey)

# ── Backend ────────────────────────────────────────────────────────────────
try:
    from train import TICKERS_TO_TRAIN
except ImportError:
    TICKERS_TO_TRAIN = ["AAPL","GOOG","MSFT","AMZN","TSLA","NVDA","AMD","NFLX"]

try:
    from backend.dynamic_trainer import train_and_predict_dynamic
    from backend.model_manager   import get_suggestion_metrics
    from backend.data_manager    import get_processed_data, FEATURE_GROUPS
    print("[app] Backend modules loaded OK")
except ImportError as e:
    print(f"[app] Backend import error: {e}")
    traceback.print_exc()
    def train_and_predict_dynamic(*a, **kw): return None, None
    def get_suggestion_metrics(*a, **kw):    return None
    def get_processed_data(*a, **kw):        return None
    FEATURE_GROUPS = {}

try:
    from backend.backtester import run_backtest
    print("[app] Backtester module loaded OK")
except ImportError as e:
    print(f"[app] Backtester import error: {e}")
    def run_backtest(*a, **kw): return None

# Valid ticker and feature group lists for input validation
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

# SECRET_KEY: generate safe random key if env var doesn't exist (invalidates sessions on restart)
_secret = os.environ.get("SECRET_KEY")
if not _secret:
    _secret = secrets.token_hex(32)
    print("[app] WARNING: SECRET_KEY env var missing! Add to Render > Environment. "
          "Temporary random key generated (sessions reset on restart).")
app.config["SECRET_KEY"] = _secret

# ── DB ──────────────────────────────────────────────────────────────────────
_db_url = os.environ.get("DATABASE_URL", "")
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)

if _db_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = _db_url
    print("[app] Using PostgreSQL")
else:
    _inst = os.path.join(BASE_DIR, "instance")
    os.makedirs(_inst, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{_inst}/fintap.db"
    print("[app] Using SQLite (Resets on every deploy in Render!)")

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle":  280,
}
app.config["WTF_CSRF_TIME_LIMIT"] = 3600   # CSRF token valid for 1 hour

# ── Flask-Mail ────────────────────────────────────────────────────────────────
app.config["MAIL_SERVER"]   = os.environ.get("MAIL_SERVER", "")
app.config["MAIL_PORT"]     = int(os.environ.get("MAIL_PORT", 587))
app.config["MAIL_USE_TLS"]  = os.environ.get("MAIL_USE_TLS", "true").lower() == "true"
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD", "")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER", "noreply@fintap.app")
mail = Mail(app)

# ── Cookie Security ────────────────────────────────────────────────────────
_is_prod = not app.debug
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = _is_prod   # HTTPS required in prod

# ── CORS ──────────────────────────────────────────────────────────────────
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
if _allowed_origins == "*":
    print("[app] WARNING: CORS is open to all origins. Set ALLOWED_ORIGINS env var in prod.")
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

# Create DB tables
with app.app_context():
    try:
        db.create_all()
        print("[app] DB tables ready OK")
    except Exception as e:
        print(f"[app] db.create_all error: {e}")

    # V0.7 migration: add accuracy_pct column to prediction table if not exists
    try:
        db.session.execute(db.text("ALTER TABLE prediction ADD COLUMN accuracy_pct FLOAT"))
        db.session.commit()
        print("[app] Migration: prediction.accuracy_pct column added OK")
    except Exception:
        db.session.rollback()
        # Column already exists, no issue


# ── Inject CSRF token to every response (for JS fetch) ──────────────────────
@app.after_request
def inject_csrf_token(response):
    response.set_cookie(
        "csrf_token", generate_csrf(),
        httponly=False,          # Must be readable by JS
        samesite="Strict",
        secure=_is_prod,
    )
    return response


# ── Security Headers ───────────────────────────────────────────────────
@app.after_request
def set_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    if _is_prod:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    return response


# ── Helpers ───────────────────────────────────────────────────────────────
def get_wallet():
    w = Wallet.query.filter_by(user_id=current_user.id).first()
    if not w:
        w = Wallet(user_id=current_user.id, balance=5)
        db.session.add(w); db.session.commit()
    return w


def _admin_check() -> bool:
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    if not admin_secret:
        return False   # Deny if no env var - never open in dev mode
    # Header priority; URL query param is also supported for CLI ease
    provided = (request.headers.get("X-Admin-Secret", "")
                or request.args.get("secret", ""))
    return secrets.compare_digest(provided, admin_secret)


def _is_admin() -> bool:
    if not current_user.is_authenticated:
        return False
    admin_emails_raw = os.environ.get("ADMIN_EMAIL", "")
    if admin_emails_raw:
        admin_emails = {e.strip().lower() for e in admin_emails_raw.split(",") if e.strip()}
        return current_user.email.lower() in admin_emails
    # Fallback: ADMIN_SECRET header
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    if not admin_secret:
        return False
    provided = request.headers.get("X-Admin-Secret", "") or request.args.get("secret", "")
    return secrets.compare_digest(provided, admin_secret)


def _is_safe_redirect_url(url: str) -> bool:
    """Open redirect protection: does the URL point to the same host?"""
    if not url:
        return False
    try:
        ref  = urlparse(request.host_url)
        test = urlparse(urljoin(request.host_url, url))
        return test.scheme in ("http", "https") and ref.netloc == test.netloc
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/ping")
def ping():
    """UptimeRobot pings this endpoint every 5 minutes → prevents Render sleep."""
    return "OK", 200


# ── Cache & Data update endpoints ─────────────────────────────────────

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
    if not _is_admin():
        return jsonify({"error": "Unauthorized access"}), 403
    ticker = request.args.get("ticker")
    if ticker and ticker not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker"}), 400
    try:
        from backend.data_manager import cache_clear
        cache_clear(ticker)
        return jsonify({"status": "ok", "cleared": ticker or "all"})
    except Exception as e:
        return jsonify({"error": "Operation failed"}), 500


@app.route("/api/refresh")
@limiter.limit("5 per hour")
def api_refresh():
    secret   = request.args.get("secret", "")
    expected = os.environ.get("CRON_SECRET", "")

    if not expected or not secrets.compare_digest(secret, expected):
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
                    print(f"[refresh] {ticker}: OK, last: {df.index[-1].date()}")
                else:
                    results["failed"].append(ticker)
            except Exception as e:
                print(f"[refresh] {ticker}: ERROR: {e}")
                results["failed"].append(ticker)

        results["skipped"] = TICKERS_TO_TRAIN[10:]
        print(f"[refresh] Completed: {len(results['refreshed'])} OK, "
              f"{len(results['failed'])} failed")

    except Exception as e:
        print(f"[refresh] ERROR: {e}"); traceback.print_exc()
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
    if not _admin_check():
        return jsonify({"error": "Unauthorized access"}), 403

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
        info["connection"] = f"ERROR: {str(e)[:200]}"
    return jsonify(info), 200


@app.route("/db-kur")
def db_kur():
    if not _admin_check():
        return jsonify({"error": "Unauthorized access"}), 403

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
                    print("[db-kur] password column extended to 512")
                except Exception as alter_e:
                    db.session.rollback()
                    print(f"[db-kur] ALTER already done or error: {alter_e}")

            # V0.7 migration: accuracy_pct column (for PostgreSQL and SQLite)
            try:
                db.session.execute(db.text(
                    "ALTER TABLE prediction ADD COLUMN accuracy_pct FLOAT"
                ))
                db.session.commit()
                print("[db-kur] prediction.accuracy_pct column added")
            except Exception as alter_e:
                db.session.rollback()
                print(f"[db-kur] accuracy_pct already exists or error: {alter_e}")

        return "Database setup and updated ✓", 200
    except Exception as e:
        return f"Error: {e}", 500


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
    wl_symbols = {item.symbol for item in Watchlist.query.filter_by(user_id=current_user.id).all()}
    return render_template("all_stocks.html", user=current_user,
                           balance=w.balance, stocks=TICKERS_TO_TRAIN, watchlist_symbols=wl_symbols)


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
    # Accuracy stats
    scored = [p for p in prs if p.accuracy_pct is not None]
    avg_acc = round(sum(p.accuracy_pct for p in scored) / len(scored), 1) if scored else None
    return render_template("profile.html", user=current_user, wallet=w,
                           balance=w.balance, transactions=txs, predictions=prs,
                           avg_accuracy=avg_acc, scored_count=len(scored))


@app.route("/portfolio")
@login_required
def portfolio():
    w = get_wallet()
    items = Watchlist.query.filter_by(user_id=current_user.id)\
                    .order_by(Watchlist.added_at.desc()).all()
    return render_template("portfolio.html", user=current_user,
                           balance=w.balance, watchlist=items,
                           all_stocks=TICKERS_TO_TRAIN)


@app.route("/admin")
@login_required
def admin_panel():
    if not _is_admin():
        return redirect(url_for("root"))
    try:
        user_count  = db.session.query(_func.count(User.id)).scalar() or 0
        pred_count  = db.session.query(_func.count(Prediction.id)).scalar() or 0
        revenue     = db.session.query(_func.sum(Transaction.amount_paid)).scalar() or 0
        token_sales = db.session.query(_func.sum(Transaction.tokens_added)).scalar() or 0
        today       = _dt.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        preds_today = Prediction.query.filter(Prediction.created_at >= today).count()
        recent_preds = (Prediction.query
                        .order_by(Prediction.created_at.desc()).limit(20).all())
        top_users    = (db.session.query(User, _func.count(Prediction.id).label("pc"))
                        .outerjoin(Prediction, User.id == Prediction.user_id)
                        .group_by(User.id)
                        .order_by(_func.count(Prediction.id).desc())
                        .limit(10).all())
        try:
            from backend.data_manager import cache_status
            cache_info = cache_status()
        except Exception:
            cache_info = {}
    except Exception as e:
        traceback.print_exc()
        return f"<pre>Admin panel error: {e}</pre>", 500

    return render_template("admin.html",
        user_count=user_count, pred_count=pred_count,
        revenue=round(float(revenue), 2), token_sales=int(token_sales),
        preds_today=preds_today, recent_preds=recent_preds,
        top_users=top_users, cache_info=cache_info)


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
    # Ticker whitelist check
    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker"}), 400
    try:
        df = get_processed_data(ticker)
        if df is None or df.empty:
            return jsonify({"error": "Data could not be retrieved"}), 404
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
        return jsonify({"error": "Missing parameter"}), 400
    if t1 not in VALID_TICKERS or t2 not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker"}), 400

    w = get_wallet()
    if w.balance <= 0:
        return jsonify({"error": "Insufficient balance"}), 402

    try:
        m1 = get_suggestion_metrics(t1)
        m2 = get_suggestion_metrics(t2)
    except Exception as e:
        print(f"[api_compare]: {e}"); traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    if not m1 or not m2:
        return jsonify({"error": "Prediction failed — try the LINEAR model"}), 500

    w.balance -= 1
    w.last_updated = __import__("datetime").datetime.utcnow()
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"[api_compare] DB record error: {e}")

    return jsonify({
        "balance": w.balance,
        t1: {"price": m1["last_price"], "predicted": m1["predicted_price"],
             "gain":  m1["potential_gain_pct"], "rsi": m1["rsi"]},
        t2: {"price": m2["last_price"], "predicted": m2["predicted_price"],
             "gain":  m2["potential_gain_pct"], "rsi": m2["rsi"]},
    })


VALID_MODELS = {"LINEAR","RANDOM_FOREST","EXTRA_TREES","GRADIENT_BOOST",
                "XGBOOST","LIGHTGBM","LSTM"}

# ── Stripe ─────────────────────────────────────────────────────────────────
TOKEN_PACKS = {
    "starter":  {"tokens": 20,  "price_cents": 299,  "name": "Starter Pack  (20 Tokens)"},
    "explorer": {"tokens": 50,  "price_cents": 599,  "name": "Explorer Pack (50 Tokens)"},
    "pro":      {"tokens": 100, "price_cents": 999,  "name": "Pro Pack      (100 Tokens)"},
    "whale":    {"tokens": 500, "price_cents": 3999, "name": "Whale Pack    (500 Tokens)"},
}

SUBSCRIPTION_PLANS = {
    "pro_monthly": {
        "name":        "Fin-TAP Professional (Monthly)",
        "price_cents": 1500,
        "tokens":      200,   # tokens added on each billing cycle
        "interval":    "month",
        "label":       "Pro Monthly",
    },
    "enterprise_monthly": {
        "name":        "Fin-TAP Enterprise (Monthly)",
        "price_cents": 4999,
        "tokens":      999,
        "interval":    "month",
        "label":       "Enterprise Monthly",
    },
}

try:
    import stripe as _stripe_lib
    _stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")
    _stripe_lib.api_key = _stripe_key
    _stripe_ok = bool(_stripe_key)
    stripe = _stripe_lib
    print(f"[app] Stripe {'active' if _stripe_ok else 'STRIPE_SECRET_KEY missing — set env var for test/prod'}")
except ImportError:
    stripe = None
    _stripe_ok = False
    print("[app] stripe package not installed — run 'pip install stripe'")


@app.route("/api/predict_run", methods=["POST"])
@login_required
@limiter.limit("10 per hour")
@csrf.exempt
def api_predict_run():
    payload  = request.get_json(silent=True) or {}
    ticker   = str(payload.get("ticker", "AAPL")).upper().strip()
    model    = str(payload.get("model",  "LINEAR")).upper().strip()
    features = payload.get("features", [])
    horizon  = int(payload.get("horizon", 14))

    # Input validation
    if ticker not in VALID_TICKERS:
        return jsonify({"error": f"Invalid ticker: {ticker}"}), 400
    if model not in VALID_MODELS:
        return jsonify({"error": f"Invalid model: {model}"}), 400
    if not isinstance(features, list):
        return jsonify({"error": "features must be a list"}), 400
    invalid_feats = [f for f in features if f not in VALID_FEATURE_GROUPS]
    if invalid_feats:
        return jsonify({"error": f"Invalid feature group(s): {invalid_feats}"}), 400
    if horizon not in {7, 14, 30, 90}:
        return jsonify({"error": "Invalid horizon. Must be 7, 14, 30, or 90."}), 400

    w = get_wallet()
    if w.balance <= 0:
        return jsonify({"error": "Insufficient balance"}), 402

    print(f"[predict_run] {ticker} | {model} | {len(features)} feature | {horizon}d")

    try:
        preds, chart_data = train_and_predict_dynamic(ticker, model, features, horizon)
    except Exception as e:
        print(f"[predict_run] EXCEPTION: {e}"); traceback.print_exc()
        return jsonify({"error": f"Model exception: {str(e)[:300]}"}), 500

    if not preds:
        return jsonify({
            "error": (
                "Prediction failed. "
                "Suggestion: Choose the LINEAR model and fewer features, "
                "or try a different stock."
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
        print(f"[predict_run] DB record error: {e}")

    return jsonify({
        "status":     "success",
        "balance":    w.balance,
        "prediction": round(float(preds[-1]), 2),
        "chart_data": chart_data,
    })


# ── Sentiment API ──────────────────────────────────────────────────────────

@app.route("/api/sentiment/<ticker>")
@login_required
@limiter.limit("30 per minute")
@csrf.exempt
def api_sentiment(ticker):
    """Yahoo Finance RSS + VADER sentiment analysis."""
    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker"}), 400
    try:
        import feedparser
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        feed_ticker = ticker.replace("-USD", "")   # BTC-USD → BTC
        url  = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                f"?s={feed_ticker}&region=US&lang=en-US")
        feed = feedparser.parse(url)

        analyzer = SentimentIntensityAnalyzer()
        articles, scores = [], []

        for entry in feed.entries[:8]:
            title   = (entry.get("title") or "")[:140]
            summary = (entry.get("summary") or entry.get("description") or "")[:200]
            text    = f"{title}. {summary}" if summary else title
            vs      = analyzer.polarity_scores(text)
            c       = vs["compound"]
            scores.append(c)

            pub = entry.get("published", "")
            articles.append({
                "title":     title,
                "sentiment": "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral"),
                "score":     round(c, 3),
                "published": pub[:16] if pub else "",
                "link":      entry.get("link", ""),
            })

        avg = sum(scores) / len(scores) if scores else 0
        overall = "BULLISH" if avg >= 0.05 else ("BEARISH" if avg <= -0.05 else "NEUTRAL")

        return jsonify({
            "ticker":   ticker,
            "overall":  overall,
            "score":    round(avg, 3),
            "articles": articles,
            "count":    len(articles),
        })
    except ImportError:
        return jsonify({"error": "packages_missing"}), 503
    except Exception as e:
        print(f"[sentiment] {ticker}: {e}")
        return jsonify({"error": "unavailable"}), 500


# ── OHLC / Candlestick API ─────────────────────────────────────────────────

# ---- AI Insight helpers ----------------------------------------------------

def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _technical_snapshot(ticker: str) -> dict | None:
    """Small deterministic analysis layer used by the AI insight endpoints."""
    try:
        df = get_processed_data(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        df = df.dropna(subset=["Close"]).copy()
        if len(df) < 30:
            return None

        closes = df["Close"].astype(float)
        current = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else current
        change_1d = ((current - prev) / prev * 100) if prev else 0.0
        change_20d = ((current - closes.iloc[-21]) / closes.iloc[-21] * 100) if len(closes) > 21 and closes.iloc[-21] else 0.0
        change_60d = ((current - closes.iloc[-61]) / closes.iloc[-61] * 100) if len(closes) > 61 and closes.iloc[-61] else change_20d

        sma20 = float(closes.tail(20).mean())
        sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else sma20
        vol20 = float(closes.pct_change().tail(20).std() * (252 ** 0.5) * 100)
        if not math.isfinite(vol20):
            vol20 = 0.0

        diffs = closes.diff().dropna().tail(14)
        gains = diffs.clip(lower=0).mean()
        losses = (-diffs.clip(upper=0)).mean()
        rs = gains / losses if losses else 99.0
        rsi = 100 - (100 / (1 + rs))
        if not math.isfinite(float(rsi)):
            rsi = 50.0

        if current > sma20 > sma50:
            trend = "bullish"
        elif current < sma20 < sma50:
            trend = "bearish"
        else:
            trend = "mixed"

        if rsi >= 70:
            rsi_state = "overbought"
        elif rsi <= 30:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"

        return {
            "ticker": ticker,
            "price": round(current, 4),
            "change_1d": round(change_1d, 2),
            "change_20d": round(change_20d, 2),
            "change_60d": round(change_60d, 2),
            "sma20": round(sma20, 4),
            "sma50": round(sma50, 4),
            "rsi": round(float(rsi), 2),
            "rsi_state": rsi_state,
            "trend": trend,
            "volatility": round(vol20, 2),
        }
    except Exception as e:
        print(f"[ai snapshot] {ticker}: {e}")
        return None


def _risk_level(snapshot: dict, expected_return: float | None = None) -> str:
    score = 0
    if snapshot.get("volatility", 0) >= 55:
        score += 2
    elif snapshot.get("volatility", 0) >= 35:
        score += 1
    if snapshot.get("rsi_state") in {"overbought", "oversold"}:
        score += 1
    if snapshot.get("trend") == "mixed":
        score += 1
    if expected_return is not None and abs(expected_return) >= 12:
        score += 1
    if score >= 3:
        return "HIGH"
    if score >= 1:
        return "MEDIUM"
    return "LOW"


def _score_snapshot(snapshot: dict, profile: str = "balanced") -> float:
    trend_score = {"bullish": 28, "mixed": 10, "bearish": -18}.get(snapshot["trend"], 0)
    momentum = max(min(snapshot["change_20d"], 25), -25)
    rsi = snapshot["rsi"]
    rsi_score = 12 if 45 <= rsi <= 62 else 6 if 35 <= rsi < 45 or 62 < rsi <= 70 else -10
    vol_penalty = min(snapshot["volatility"] / 3, 22)
    score = 50 + trend_score + momentum + rsi_score - vol_penalty
    if profile == "low_risk":
        score -= min(snapshot["volatility"] / 2, 28)
    elif profile == "momentum":
        score += max(min(snapshot["change_60d"], 30), -20) * 0.6
    return round(max(0, min(100, score)), 1)


@app.route("/api/ai/analyst", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
@csrf.exempt
def api_ai_analyst():
    payload = request.get_json(silent=True) or {}
    ticker = str(payload.get("ticker", "AAPL")).upper().strip()
    prediction = _safe_float(payload.get("prediction"))
    horizon = int(payload.get("horizon", 14) or 14)

    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker"}), 400

    snap = _technical_snapshot(ticker)
    if not snap:
        return jsonify({"error": "Analysis data is not ready"}), 422

    expected_return = None
    if prediction and snap["price"]:
        expected_return = round((prediction - snap["price"]) / snap["price"] * 100, 2)

    direction = "upside" if (expected_return or 0) > 1 else "downside" if (expected_return or 0) < -1 else "flat"
    risk = _risk_level(snap, expected_return)
    confidence = "HIGH" if snap["trend"] != "mixed" and risk != "HIGH" else "MEDIUM" if risk != "HIGH" else "LOW"

    if direction == "upside":
        summary = f"{ticker} shows a positive {horizon}-day setup, but confirmation should come from trend and risk controls."
    elif direction == "downside":
        summary = f"{ticker} has negative pressure in the {horizon}-day forecast, so capital protection matters more than entry timing."
    else:
        summary = f"{ticker} is close to neutral for the {horizon}-day horizon; the setup needs a cleaner signal."

    bullets = [
        f"Trend is {snap['trend']} with price near ${snap['price']}.",
        f"RSI is {snap['rsi']} ({snap['rsi_state']}), which shapes short-term timing risk.",
        f"20-day move is {snap['change_20d']}% and annualized volatility is about {snap['volatility']}%.",
    ]
    if expected_return is not None:
        bullets.insert(0, f"Model-implied return potential is {expected_return:+.2f}%.")

    return jsonify({
        "ticker": ticker,
        "summary": summary,
        "bullets": bullets,
        "risk": risk,
        "confidence": confidence,
        "expected_return": expected_return,
        "disclaimer": "Educational analysis only, not financial advice.",
        "snapshot": snap,
    })


@app.route("/api/ai/portfolio")
@login_required
@limiter.limit("20 per minute")
@csrf.exempt
def api_ai_portfolio():
    items = Watchlist.query.filter_by(user_id=current_user.id).order_by(Watchlist.added_at.desc()).all()
    snapshots = [s for s in (_technical_snapshot(item.symbol) for item in items) if s]

    if not snapshots:
        return jsonify({
            "summary": "Add instruments to your watchlist to unlock AI portfolio insight.",
            "best": None,
            "riskiest": None,
            "actions": [],
            "items": [],
        })

    scored = []
    for snap in snapshots:
        score = _score_snapshot(snap)
        scored.append({**snap, "score": score, "risk": _risk_level(snap)})

    best = max(scored, key=lambda x: x["score"])
    riskiest = max(scored, key=lambda x: (x["volatility"], abs(x["change_20d"])))
    bullish_count = sum(1 for s in scored if s["trend"] == "bullish")
    high_risk_count = sum(1 for s in scored if s["risk"] == "HIGH")

    actions = []
    if high_risk_count:
        actions.append(f"Review position sizing on {high_risk_count} high-risk instrument(s).")
    actions.append(f"Watch {best['ticker']} for continuation if price stays above SMA20.")
    if riskiest["ticker"] != best["ticker"]:
        actions.append(f"Use tighter alerts on {riskiest['ticker']} because volatility is {riskiest['volatility']}%.")

    return jsonify({
        "summary": (
            f"{bullish_count}/{len(scored)} tracked instruments are in bullish trend. "
            f"{best['ticker']} has the strongest blended score, while {riskiest['ticker']} carries the highest volatility risk."
        ),
        "best": best,
        "riskiest": riskiest,
        "actions": actions,
        "items": sorted(scored, key=lambda x: x["score"], reverse=True),
    })


@app.route("/api/ai/backtest_explain", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
@csrf.exempt
def api_ai_backtest_explain():
    d = request.get_json(silent=True) or {}
    ticker = str(d.get("ticker", "")).upper().strip()
    total_return = _safe_float(d.get("total_return"), 0.0)
    bah_return = _safe_float(d.get("bah_return"), 0.0)
    win_rate = _safe_float(d.get("win_rate"), 0.0)
    drawdown = _safe_float(d.get("max_drawdown"), 0.0)
    sharpe = _safe_float(d.get("sharpe"), 0.0)
    trade_count = int(d.get("trade_count") or 0)

    edge = round(total_return - bah_return, 2)
    risk = "HIGH" if drawdown >= 20 or sharpe < 0 else "MEDIUM" if drawdown >= 10 or sharpe < 1 else "LOW"

    if edge > 3:
        summary = f"The strategy beat buy-and-hold by {edge:+.2f}% on {ticker or 'this asset'}."
    elif edge < -3:
        summary = f"The strategy lagged buy-and-hold by {edge:+.2f}%, so the rules may be too defensive for this period."
    else:
        summary = "The strategy performed close to buy-and-hold; the edge is not strong enough by itself."

    notes = [
        f"Win rate is {win_rate}% across {trade_count} trades.",
        f"Max drawdown is {drawdown}%, giving the run a {risk.lower()} risk profile.",
        f"Sharpe ratio is {sharpe}, so risk-adjusted quality is {'strong' if sharpe >= 1 else 'weak' if sharpe < 0 else 'moderate'}.",
    ]
    suggestions = []
    if trade_count < 5:
        suggestions.append("Use a longer lookback before trusting the result.")
    if drawdown >= 15:
        suggestions.append("Try shorter horizons or disable short selling to reduce drawdown.")
    if win_rate < 45:
        suggestions.append("Tighten entries with RSI or trend confirmation.")
    if not suggestions:
        suggestions.append("Compare the same setup across 30d and 90d horizons before scaling it.")

    return jsonify({
        "summary": summary,
        "risk": risk,
        "edge_vs_buy_hold": edge,
        "notes": notes,
        "suggestions": suggestions,
        "disclaimer": "Backtests are historical simulations, not guarantees.",
    })


@app.route("/api/ai/screener")
@login_required
@limiter.limit("10 per minute")
@csrf.exempt
def api_ai_screener():
    profile = request.args.get("profile", "balanced").strip().lower()
    if profile not in {"balanced", "low_risk", "momentum"}:
        profile = "balanced"

    rows = []
    for ticker in TICKERS_TO_TRAIN:
        snap = _technical_snapshot(ticker)
        if not snap:
            continue
        score = _score_snapshot(snap, profile)
        rows.append({
            **snap,
            "score": score,
            "risk": _risk_level(snap),
            "reason": f"{snap['trend']} trend, RSI {snap['rsi']}, 20d {snap['change_20d']:+.2f}%",
        })

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)[:12]
    return jsonify({
        "profile": profile,
        "summary": f"Top {len(rows)} instruments ranked by {profile.replace('_', ' ')} AI score.",
        "results": rows,
    })


@app.route("/api/ai/chat", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
@csrf.exempt
def api_ai_chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    page = str(payload.get("page", "")).strip("/")

    if not message:
        return jsonify({"error": "Message is required."}), 400
    if len(message) > 600:
        return jsonify({"error": "Message is too long. Keep it under 600 characters."}), 400

    text = message.lower()
    quick_links = []

    if any(k in text for k in ("backtest", "short", "açığa", "aciga", "strategy", "strateji")):
        answer = (
            "Backtest tests the RSI + SMA trend strategy on your chosen stock using historical data. "
            "If 'Allow Short Selling' is enabled, short transactions are also simulated on downtrend signals. "
            "In the results, you should look at strategy return, buy-and-hold, win rate, max drawdown, and Sharpe ratio values."
        )
        quick_links = [{"label": "Open Backtester", "url": "/backtest"}]
    elif any(k in text for k in ("tahmin", "forecast", "predict", "prediction", "model")):
        answer = (
            "On the AI Forecast page, you can choose the ticker, model, horizon, and technical indicators to get a price prediction. "
            "In the short term, 7 or 14 days are more balanced; 90 days is highly uncertain. "
            "After the prediction, the AI Analyst Summary provides commentaries on risk, confidence, and expected return."
        )
        quick_links = [{"label": "Open AI Forecast", "url": "/predict"}]
    elif any(k in text for k in ("portfolio", "watchlist", "portföy", "portfoy", "takip", "track")):
        answer = (
            "You can track your watchlist instruments on the Portfolio page. "
            "The AI Portfolio Insight card summarizes the strongest blended score, the riskiest asset, and tracking actions in your list."
        )
        quick_links = [{"label": "Open Portfolio", "url": "/portfolio"}]
    elif any(k in text for k in ("screener", "hisse bul", "listele", "öner", "oner", "recommend", "find")):
        answer = (
            "The AI Stock Screener on the All Stocks page ranks instruments according to Balanced, Low Risk, or Momentum profiles. "
            "The score is calculated based on a combination of trend, RSI, recent movement, and volatility."
        )
        quick_links = [{"label": "Open Screener", "url": "/all_stocks"}]
    elif any(k in text for k in ("token", "bakiye", "ödeme", "odeme", "pricing", "price", "balance", "pay")):
        answer = (
            "Running a prediction consumes tokens. You can see your current token balance at the top right. "
            "When your tokens are running low, you can buy packs or subscriptions from the Pricing page."
        )
        quick_links = [{"label": "Open Pricing", "url": "/prices"}]
    elif any(k in text for k in ("risk", "rsi", "drawdown", "sharpe", "volatil", "volatile")):
        answer = (
            "RSI alone is not enough to read risk properly. RSI, trend, volatility, max drawdown, and Sharpe ratio should be interpreted together. "
            "As drawdown grows, capital risk increases; if the Sharpe ratio is above 1, risk-adjusted performance is considered healthier."
        )
    else:
        page_hint = f" You are currently on the /{page} page; you can ask a more specific question about the tool here." if page else ""
        answer = (
            "I am the Fin-TAP assistant. I can help you with forecast, backtest, portfolio, screener, tokens, and risk metrics."
            f"{page_hint} Example: 'How do I interpret backtest results?' or 'What does allow short selling do?'"
        )

    return jsonify({
        "answer": answer,
        "quick_links": quick_links,
        "disclaimer": "Educational assistant only, not financial advice.",
    })


@app.route("/api/ohlc/<ticker>")
@login_required
@limiter.limit("60 per minute")
@csrf.exempt
def api_ohlc(ticker):
    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker"}), 400

    try:
        import numpy as np

        df = get_processed_data(ticker)
        if df is None or df.empty:
            return jsonify({"error": "No data"}), 422

        df = df.tail(120).copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        closes  = df["Close"].values.astype(float)
        highs   = df["High"].values.astype(float)
        lows    = df["Low"].values.astype(float)
        opens   = df["Open"].values.astype(float)
        volumes = (df["Volume"].values.astype(float)
                   if "Volume" in df.columns else [0.0] * len(closes))
        dates   = [d.strftime("%Y-%m-%d") for d in df.index]
        n       = len(closes)

        # ── Candles ──
        candles = [
            {"time": dates[i], "open": round(float(opens[i]), 4),
             "high": round(float(highs[i]), 4), "low": round(float(lows[i]), 4),
             "close": round(float(closes[i]), 4)}
            for i in range(n)
        ]

        # ── SMA ──
        def sma_series(period):
            out = []
            for i in range(n):
                if i + 1 < period:
                    continue
                val = float(np.mean(closes[i + 1 - period: i + 1]))
                out.append({"time": dates[i], "value": round(val, 4)})
            return out

        sma20 = sma_series(20)
        sma50 = sma_series(50)

        # ── Bollinger Bands (20, 2σ) ──
        bb_upper, bb_lower, bb_mid = [], [], []
        for i in range(19, n):
            window = closes[i - 19: i + 1]
            mid    = float(np.mean(window))
            std    = float(np.std(window, ddof=1))
            bb_mid.append({"time": dates[i],   "value": round(mid, 4)})
            bb_upper.append({"time": dates[i], "value": round(mid + 2 * std, 4)})
            bb_lower.append({"time": dates[i], "value": round(mid - 2 * std, 4)})

        # ── RSI(14) ──
        rsi_series = []
        for i in range(14, n):
            diffs  = np.diff(closes[i - 14: i + 1])
            gains  = np.where(diffs > 0, diffs, 0.0)
            losses = np.where(diffs < 0, -diffs, 0.0)
            avg_g  = float(np.mean(gains))
            avg_l  = float(np.mean(losses)) if np.any(losses) else 1e-9
            rsi_val = 100 - 100 / (1 + avg_g / avg_l)
            rsi_series.append({"time": dates[i], "value": round(rsi_val, 2)})

        # ── Volume (normalised for bar chart) ──
        vol_series = [
            {"time": dates[i], "value": int(volumes[i]),
             "color": "rgba(0,230,118,0.4)" if closes[i] >= opens[i] else "rgba(255,69,96,0.4)"}
            for i in range(n)
        ]

        return jsonify({
            "ticker":   ticker,
            "candles":  candles,
            "sma20":    sma20,
            "sma50":    sma50,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_mid":   bb_mid,
            "rsi":      rsi_series,
            "volume":   vol_series,
        })

    except Exception as e:
        print(f"[ohlc] {ticker}: {e}")
        return jsonify({"error": "unavailable"}), 500


# ── Watchlist API ──────────────────────────────────────────────────────────

@app.route("/api/watchlist", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
@csrf.exempt
def api_watchlist_get():
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
    payload = request.get_json(silent=True) or {}
    symbol  = str(payload.get("symbol", "")).upper().strip()

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    if symbol not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker"}), 400

    existing = Watchlist.query.filter_by(
        user_id=current_user.id, symbol=symbol
    ).first()
    if existing:
        return jsonify({"status": "already_exists", "symbol": symbol})

    wl_count = Watchlist.query.filter_by(user_id=current_user.id).count()
    if wl_count >= 30:
        return jsonify({"error": "The watchlist can contain a maximum of 30 instruments."}), 400

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
    """Remove a stock from the watchlist."""
    symbol = symbol.upper().strip()
    item   = Watchlist.query.filter_by(
        user_id=current_user.id, symbol=symbol
    ).first()
    if not item:
        return jsonify({"error": "Not found"}), 404
    db.session.delete(item)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "removed", "symbol": symbol})


# ── Price Alerts API ───────────────────────────────────────────────────────

MAX_ALERTS_PER_USER = 20


@app.route("/alerts")
@login_required
def alerts_page():
    w = get_wallet()
    return render_template("alerts.html",
                           trained_stocks=TICKERS_TO_TRAIN,
                           balance=w.balance,
                           user=current_user)


@app.route("/api/alerts", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
@csrf.exempt
def api_alerts_list():
    alerts = (PriceAlert.query
              .filter_by(user_id=current_user.id)
              .order_by(PriceAlert.created_at.desc())
              .all())
    result = []
    for a in alerts:
        result.append({
            "id":           a.id,
            "symbol":       a.symbol,
            "target_price": a.target_price,
            "direction":    a.direction,
            "note":         a.note or "",
            "status":       a.status,
            "created_at":   a.created_at.strftime("%Y-%m-%d %H:%M"),
            "triggered_at": a.triggered_at.strftime("%Y-%m-%d %H:%M") if a.triggered_at else None,
        })
    return jsonify(result)


@app.route("/api/alerts", methods=["POST"])
@login_required
@limiter.limit("30 per hour")
@csrf.exempt
def api_alerts_create():
    payload      = request.get_json(silent=True) or {}
    symbol       = str(payload.get("symbol", "")).upper().strip()
    direction    = str(payload.get("direction", "")).lower().strip()
    target_price = payload.get("target_price")
    note         = str(payload.get("note", ""))[:200].strip()

    if symbol not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker."}), 400
    if direction not in ("above", "below"):
        return jsonify({"error": "direction must be 'above' or 'below'."}), 400
    try:
        target_price = float(target_price)
        if target_price <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid target price."}), 400

    active_count = PriceAlert.query.filter_by(
        user_id=current_user.id, status="active"
    ).count()
    if active_count >= MAX_ALERTS_PER_USER:
        return jsonify({"error": f"You can create a maximum of {MAX_ALERTS_PER_USER} active alerts."}), 400

    alert = PriceAlert(
        user_id=current_user.id,
        symbol=symbol,
        target_price=round(target_price, 6),
        direction=direction,
        note=note or None,
        status="active",
    )
    db.session.add(alert)
    db.session.commit()
    return jsonify({"status": "created", "id": alert.id}), 201


@app.route("/api/alerts/<int:alert_id>", methods=["DELETE"])
@login_required
@limiter.limit("30 per hour")
@csrf.exempt
def api_alerts_delete(alert_id):
    alert = PriceAlert.query.filter_by(
        id=alert_id, user_id=current_user.id
    ).first_or_404()
    db.session.delete(alert)
    db.session.commit()
    return jsonify({"status": "deleted"})


@app.route("/api/alerts/<int:alert_id>/cancel", methods=["POST"])
@login_required
@limiter.limit("30 per hour")
@csrf.exempt
def api_alerts_cancel(alert_id):
    alert = PriceAlert.query.filter_by(
        id=alert_id, user_id=current_user.id
    ).first_or_404()
    alert.status = "cancelled"
    db.session.commit()
    return jsonify({"status": "cancelled"})


@app.route("/api/alerts/check")
@csrf.exempt
def api_alerts_check():
    expected = os.environ.get("CRON_SECRET", "")
    if not expected:
        return jsonify({"error": "CRON_SECRET not configured"}), 503

    provided = request.headers.get("X-Cron-Secret", "") or request.args.get("secret", "")
    if not secrets.compare_digest(expected, provided):
        return jsonify({"error": "Unauthorized"}), 401

    active_alerts = PriceAlert.query.filter_by(status="active").all()
    if not active_alerts:
        return jsonify({"checked": 0, "triggered": 0})

    # Fetch prices by symbol once
    price_cache: dict[str, float] = {}
    for alert in active_alerts:
        if alert.symbol not in price_cache:
            try:
                df = get_processed_data(alert.symbol)
                if df is not None and not df.empty:
                    price_cache[alert.symbol] = float(df["Close"].iloc[-1])
            except Exception:
                pass

    triggered_count = 0
    for alert in active_alerts:
        current_price = price_cache.get(alert.symbol)
        if current_price is None:
            continue

        fired = (
            (alert.direction == "above" and current_price >= alert.target_price) or
            (alert.direction == "below" and current_price <= alert.target_price)
        )
        if not fired:
            continue

        # Triggered — update status
        alert.status       = "triggered"
        alert.triggered_at = _dt.utcnow()
        triggered_count   += 1

        # Send email
        try:
            user = db.session.get(User, alert.user_id)
            if user and app.config.get("MAIL_SERVER"):
                direction_word = "risen above" if alert.direction == "above" else "fallen below"
                subject = f"[Fin-TAP] Price Alert: {alert.symbol} {direction_word} ${alert.target_price:,.2f}"
                body = (
                    f"Hi {user.name},\n\n"
                    f"Your price alert for {alert.symbol} has been triggered.\n\n"
                    f"  Alert:         {alert.symbol} {direction_word} ${alert.target_price:,.2f}\n"
                    f"  Current Price: ${current_price:,.2f}\n"
                )
                if alert.note:
                    body += f"  Your note:     {alert.note}\n"
                body += (
                    f"\n  Triggered at:  {alert.triggered_at.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"Log in to Fin-TAP to manage your alerts.\n\n"
                    f"— Fin-TAP Team\n"
                )
                msg = Message(subject=subject, recipients=[user.email], body=body)
                mail.send(msg)
                print(f"[alerts] Email sent → {user.email} ({alert.symbol})")
            else:
                print(f"[alerts] MAIL_SERVER missing — alert triggered but email not sent: {alert.symbol}")
        except Exception as e:
            print(f"[alerts] Email error: {e}")

    db.session.commit()
    print(f"[alerts] Check complete: {len(active_alerts)} checked, {triggered_count} triggered")
    return jsonify({"checked": len(active_alerts), "triggered": triggered_count})


# ── Paper Trading API ──────────────────────────────────────────────────────

def _get_or_create_paper(user_id: int) -> PaperPortfolio:
    p = PaperPortfolio.query.filter_by(user_id=user_id).first()
    if not p:
        p = PaperPortfolio(user_id=user_id, cash=PAPER_STARTING_CASH)
        db.session.add(p)
        db.session.commit()
    return p


@app.route("/paper")
@login_required
def paper_page():
    w = get_wallet()
    return render_template("paper_trading.html",
                           trained_stocks=TICKERS_TO_TRAIN,
                           balance=w.balance,
                           user=current_user)


@app.route("/api/paper/portfolio")
@login_required
@limiter.limit("60 per minute")
@csrf.exempt
def api_paper_portfolio():
    port = _get_or_create_paper(current_user.id)
    positions = PaperPosition.query.filter_by(user_id=current_user.id).all()

    pos_data = []
    total_market_value = 0.0

    for pos in positions:
        if pos.quantity <= 0:
            continue
        try:
            df = get_processed_data(pos.symbol)
            cur_price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
        except Exception:
            cur_price = None

        market_val = round(pos.quantity * cur_price, 2) if cur_price else None
        cost_basis = round(pos.quantity * pos.avg_cost, 2)
        pnl        = round(market_val - cost_basis, 2) if market_val is not None else None
        pnl_pct    = round((pnl / cost_basis) * 100, 2) if (pnl is not None and cost_basis) else None

        if market_val:
            total_market_value += market_val

        pos_data.append({
            "symbol":       pos.symbol,
            "quantity":     round(pos.quantity, 6),
            "avg_cost":     round(pos.avg_cost, 4),
            "cur_price":    round(cur_price, 4) if cur_price else None,
            "market_value": market_val,
            "cost_basis":   cost_basis,
            "pnl":          pnl,
            "pnl_pct":      pnl_pct,
        })

    total_equity = round(port.cash + total_market_value, 2)
    total_return = round(total_equity - PAPER_STARTING_CASH, 2)
    total_return_pct = round((total_return / PAPER_STARTING_CASH) * 100, 2)

    return jsonify({
        "cash":             round(port.cash, 2),
        "market_value":     round(total_market_value, 2),
        "total_equity":     total_equity,
        "total_return":     total_return,
        "total_return_pct": total_return_pct,
        "starting_cash":    PAPER_STARTING_CASH,
        "positions":        sorted(pos_data, key=lambda x: -(x["market_value"] or 0)),
    })


@app.route("/api/paper/trade", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
@csrf.exempt
def api_paper_trade():
    """Simulated buy/sell transaction."""
    payload  = request.get_json(silent=True) or {}
    symbol   = str(payload.get("symbol", "")).upper().strip()
    action   = str(payload.get("action", "")).lower().strip()
    quantity = payload.get("quantity")

    if symbol not in VALID_TICKERS:
        return jsonify({"error": "Invalid ticker."}), 400
    if action not in ("buy", "sell"):
        return jsonify({"error": "action must be 'buy' or 'sell'."}), 400
    try:
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid quantity."}), 400

    # Get current price
    try:
        df = get_processed_data(symbol)
        if df is None or df.empty:
            return jsonify({"error": "Price data could not be retrieved."}), 422
        cur_price = float(df["Close"].iloc[-1])
    except Exception:
        return jsonify({"error": "Price data could not be retrieved."}), 422

    total_cost = round(quantity * cur_price, 6)
    port = _get_or_create_paper(current_user.id)

    if action == "buy":
        if port.cash < total_cost:
            return jsonify({"error": f"Insufficient balance. Available: ${port.cash:,.2f}, Required: ${total_cost:,.2f}"}), 400

        port.cash = round(port.cash - total_cost, 6)

        pos = PaperPosition.query.filter_by(
            user_id=current_user.id, symbol=symbol
        ).first()
        if pos:
            # Update average cost
            new_qty  = pos.quantity + quantity
            pos.avg_cost = round((pos.avg_cost * pos.quantity + cur_price * quantity) / new_qty, 6)
            pos.quantity = round(new_qty, 6)
        else:
            pos = PaperPosition(
                user_id=current_user.id, symbol=symbol,
                quantity=round(quantity, 6), avg_cost=round(cur_price, 6),
            )
            db.session.add(pos)

    else:  # sell
        pos = PaperPosition.query.filter_by(
            user_id=current_user.id, symbol=symbol
        ).first()
        if not pos or pos.quantity < quantity:
            held = round(pos.quantity, 4) if pos else 0
            return jsonify({"error": f"Insufficient positions. You hold: {held} {symbol}"}), 400

        port.cash    = round(port.cash + total_cost, 6)
        pos.quantity = round(pos.quantity - quantity, 6)
        if pos.quantity < 1e-6:
            db.session.delete(pos)

    trade = PaperTrade(
        user_id=current_user.id, symbol=symbol, action=action,
        quantity=round(quantity, 6), price=round(cur_price, 4),
        total=round(total_cost, 2),
    )
    db.session.add(trade)
    db.session.commit()

    return jsonify({
        "status":    "executed",
        "action":    action,
        "symbol":    symbol,
        "quantity":  round(quantity, 6),
        "price":     round(cur_price, 4),
        "total":     round(total_cost, 2),
        "new_cash":  round(port.cash, 2),
    })


@app.route("/api/paper/history")
@login_required
@limiter.limit("30 per minute")
@csrf.exempt
def api_paper_history():
    """Last 50 transaction history."""
    trades = (PaperTrade.query
              .filter_by(user_id=current_user.id)
              .order_by(PaperTrade.executed_at.desc())
              .limit(50).all())
    return jsonify([{
        "id":          t.id,
        "symbol":      t.symbol,
        "action":      t.action,
        "quantity":    round(t.quantity, 6),
        "price":       round(t.price, 4),
        "total":       round(t.total, 2),
        "executed_at": t.executed_at.strftime("%Y-%m-%d %H:%M"),
    } for t in trades])


@app.route("/api/paper/reset", methods=["POST"])
@login_required
@limiter.limit("3 per hour")
@csrf.exempt
def api_paper_reset():
    """Reset virtual portfolio — clear all positions and history."""
    port = _get_or_create_paper(current_user.id)

    PaperPosition.query.filter_by(user_id=current_user.id).delete()
    PaperTrade.query.filter_by(user_id=current_user.id).delete()

    port.cash     = PAPER_STARTING_CASH
    port.reset_at = _dt.utcnow()
    db.session.commit()

    return jsonify({"status": "reset", "cash": PAPER_STARTING_CASH})


# ── Correlation Matrix API ─────────────────────────────────────────────────

@app.route("/correlation")
@login_required
def correlation_page():
    w = get_wallet()
    return render_template("correlation.html",
                           trained_stocks=TICKERS_TO_TRAIN,
                           balance=w.balance,
                           user=current_user)


@app.route("/api/correlation")
@login_required
@limiter.limit("10 per minute")
@csrf.exempt
def api_correlation():
    """
    Returns the daily log-return correlation matrix for selected tickers.
    Query params:
      tickers : comma-separated symbols (max 20)
      days    : lookback (30 | 90 | 180 | 365)
    """
    import numpy as np

    raw_tickers = request.args.get("tickers", "")
    days        = int(request.args.get("days", 90))

    if days not in (30, 90, 180, 365):
        days = 90

    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    tickers = [t for t in tickers if t in VALID_TICKERS][:20]

    if len(tickers) < 2:
        return jsonify({"error": "Select at least 2 valid tickers."}), 400

    # ── Fetch daily log-return series for each ticker ─────────────────────
    returns_map: dict[str, np.ndarray] = {}
    for ticker in tickers:
        try:
            df = get_processed_data(ticker)
            if df is None or df.empty:
                continue
            closes = df["Close"].tail(days + 1).values.astype(float)
            if len(closes) < 10:
                continue
            log_ret = np.diff(np.log(closes))
            returns_map[ticker] = log_ret
        except Exception as e:
            print(f"[corr] {ticker}: {e}")

    valid = list(returns_map.keys())
    if len(valid) < 2:
        return jsonify({"error": "Insufficient data found."}), 422

    # Trim to common length (based on shortest series)
    min_len = min(len(v) for v in returns_map.values())
    matrix_data = np.array([returns_map[t][-min_len:] for t in valid])  # shape: (n_tickers, min_len)

    # Pearson correlation matrix
    corr = np.corrcoef(matrix_data)

    # ── Summary statistics (highest / lowest correlation pairs) ──────
    pairs = []
    n = len(valid)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "a": valid[i], "b": valid[j],
                "r": round(float(corr[i, j]), 4)
            })
    pairs.sort(key=lambda x: x["r"], reverse=True)

    return jsonify({
        "tickers":  valid,
        "matrix":   [[round(float(corr[i][j]), 4) for j in range(n)] for i in range(n)],
        "days":     min_len,
        "top_pos":  pairs[:3],           # strongest positive correlation
        "top_neg":  pairs[-3:][::-1],    # strongest negative correlation
    })


# ── Prediction Accuracy API ────────────────────────────────────────────────

@app.route("/api/accuracy/update")
@login_required
@limiter.limit("5 per hour")
@csrf.exempt
def api_accuracy_update():
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
#  PAYMENT (Stripe)
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/payment/create-checkout-session", methods=["POST"])
@login_required
@limiter.limit("20 per hour")
@csrf.exempt
def api_create_checkout():
    if not _stripe_ok:
        return jsonify({"error": "Payment system is not active yet. STRIPE_SECRET_KEY env var is not set."}), 503

    payload = request.get_json(silent=True) or {}
    pack_id = str(payload.get("pack", "")).lower().strip()

    if pack_id not in TOKEN_PACKS:
        return jsonify({"error": "Invalid pack"}), 400

    pack       = TOKEN_PACKS[pack_id]
    base_url   = request.host_url.rstrip("/")
    success_url = f"{base_url}/payment/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url  = f"{base_url}/prices"

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": f"Fin-TAP {pack['name']}",
                        "description": f"{pack['tokens']} AI prediction token — Fin-TAP platform",
                    },
                    "unit_amount": pack["price_cents"],
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "user_id": str(current_user.id),
                "pack_id": pack_id,
                "tokens":  str(pack["tokens"]),
            },
        )
        print(f"[stripe] checkout session created: user={current_user.id} pack={pack_id}")
        return jsonify({"url": session.url})
    except Exception as e:
        print(f"[stripe] checkout session error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Payment could not be initiated. Please try again."}), 500


@app.route("/api/payment/create-subscription", methods=["POST"])
@login_required
@limiter.limit("10 per hour")
@csrf.exempt
def api_create_subscription():
    """Create Stripe Subscription Checkout session (monthly subscription)."""
    if not _stripe_ok:
        return jsonify({"error": "Payment system is not active yet. STRIPE_SECRET_KEY env var is not set."}), 503

    payload = request.get_json(silent=True) or {}
    plan_id = payload.get("plan", "")

    if plan_id not in SUBSCRIPTION_PLANS:
        return jsonify({"error": "Invalid subscription plan."}), 400

    plan     = SUBSCRIPTION_PLANS[plan_id]
    base_url = request.host_url.rstrip("/")
    success_url = f"{base_url}/payment/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url  = f"{base_url}/prices"

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency":    "usd",
                    "unit_amount": plan["price_cents"],
                    "product_data": {"name": plan["name"]},
                    "recurring":   {"interval": plan["interval"]},
                },
                "quantity": 1,
            }],
            mode="subscription",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "user_id": str(current_user.id),
                "plan_id": plan_id,
                "tokens":  str(plan["tokens"]),
            },
        )
        print(f"[stripe] subscription session created: user={current_user.id} plan={plan_id}")
        return jsonify({"url": session.url})
    except Exception as e:
        print(f"[stripe] subscription session error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Subscription could not be initiated. Please try again."}), 500


@app.route("/api/payment/webhook", methods=["POST"])
@csrf.exempt
def stripe_webhook():
    """
    Stripe webhook endpoint.
    Listens to checkout.session.completed event → updates token balance.
    """
    if not _stripe_ok:
        return "", 400

    payload    = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")
    wh_secret  = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    # 1) Signature verification / parse
    try:
        if wh_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, wh_secret)
        else:
            print("[stripe webhook] WARNING: STRIPE_WEBHOOK_SECRET missing, signature not verified!")
            event = json.loads(payload)
    except Exception as e:
        print(f"[stripe webhook] signature/parse error: {e}")
        return jsonify({"error": str(e)}), 400

    # 2) Parse raw payload as JSON — bypass SDK typed object problems
    try:
        raw_event  = json.loads(payload)
        event_type = raw_event.get("type", "")

        if event_type == "checkout.session.completed":
            sess_data  = raw_event["data"]["object"]
            session_id = sess_data.get("id", "")
            meta       = sess_data.get("metadata") or {}
            mode       = sess_data.get("mode", "payment")

            user_id = int(meta.get("user_id") or 0)
            tokens  = int(meta.get("tokens")  or 0)
            pack_id = str(meta.get("pack_id") or "")
            plan_id = str(meta.get("plan_id") or "")

            print(f"[stripe webhook] session={session_id} mode={mode} user={user_id} tokens={tokens}")

            if user_id and tokens:
                w = Wallet.query.filter_by(user_id=user_id).first()
                if not w:
                    w = Wallet(user_id=user_id, balance=0)
                    db.session.add(w)
                w.balance += tokens

                if mode == "subscription":
                    plan   = SUBSCRIPTION_PLANS.get(plan_id, {})
                    amount = plan.get("price_cents", 0) / 100
                else:
                    pack   = TOKEN_PACKS.get(pack_id, {})
                    amount = pack.get("price_cents", 0) / 100

                db.session.add(Transaction(user_id=user_id, amount_paid=amount, tokens_added=tokens))
                db.session.commit()
                print(f"[stripe webhook] user={user_id} +{tokens} token added OK (mode={mode})")
            else:
                print(f"[stripe webhook] WARNING: metadata empty — session={session_id} meta={meta}")

        elif event_type == "invoice.payment_succeeded":
            # Monthly renewal token loading
            invoice   = raw_event["data"]["object"]
            billing   = invoice.get("billing_reason", "")
            # Process only subscription_cycle (renewal) events; initial payment is handled via checkout.session.completed
            if billing == "subscription_cycle":
                sub_id = invoice.get("subscription", "")
                try:
                    sub    = stripe.Subscription.retrieve(sub_id)
                    meta   = sub.get("metadata") or {}
                    user_id = int(meta.get("user_id") or 0)
                    plan_id = str(meta.get("plan_id") or "")
                    tokens  = int(meta.get("tokens")  or 0)
                    amount  = (invoice.get("amount_paid") or 0) / 100
                    if user_id and tokens:
                        w = Wallet.query.filter_by(user_id=user_id).first()
                        if w:
                            w.balance += tokens
                            db.session.add(Transaction(user_id=user_id, amount_paid=amount, tokens_added=tokens))
                            db.session.commit()
                            print(f"[stripe webhook] renewal: user={user_id} +{tokens} token (plan={plan_id})")
                except Exception as sub_err:
                    print(f"[stripe webhook] subscription renewal error: {sub_err}")

    except Exception as e:
        db.session.rollback()
        print(f"[stripe webhook] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)[:200]}), 500

    return jsonify({"status": "ok"})


@app.route("/payment/success")
@login_required
def payment_success():
    w = get_wallet()
    return render_template("payment_success.html", user=current_user,
                           balance=w.balance,
                           session_id=request.args.get("session_id", ""))


# ── AUTH ────────────────────────────────────────────────═══════════════════

def _password_strong(pw: str) -> tuple[bool, str]:
    """Password strength validation. (bool, error_message)"""
    if len(pw) < 8:
        return False, "Password must be at least 8 characters long."
    has_letter = any(c.isalpha() for c in pw)
    has_digit  = any(c.isdigit() for c in pw)
    if not has_letter or not has_digit:
        return False, "Password must contain at least 1 letter and 1 number."
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
            flash("Please fill in all fields.", "error")
            return redirect(url_for("register"))

        if len(email) > 254:
            flash("Invalid email address.", "error")
            return redirect(url_for("register"))
        if len(name) > 100:
            flash("Name can be a maximum of 100 characters.", "error")
            return redirect(url_for("register"))

        ok, msg = _password_strong(password)
        if not ok:
            flash(msg, "error")
            return redirect(url_for("register"))

        # Prevent email enumeration: use the same generic error message
        if User.query.filter_by(email=email).first():
            flash("Registration could not be completed. Please try a different email.", "error")
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
            print(f"[register] New user: {email}")
            return redirect(url_for("root"))
        except Exception as e:
            db.session.rollback()
            print(f"[register] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            flash("An error occurred during registration. Please try again.", "error")
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
            next_url = request.args.get("next", "")
            return redirect(next_url if _is_safe_redirect_url(next_url) else url_for("root"))
        flash("Incorrect email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ── Password Reset ───────────────────────────────────────────────────────────

def _reset_serializer():
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="pw-reset-v1")


def _send_reset_email(user, reset_url: str):
    if not app.config.get("MAIL_SERVER"):
        return False
    try:
        msg = Message(
            subject="Fin-TAP — Password Reset",
            recipients=[user.email],
        )
        msg.body = (
            f"Hello {user.name},\n\n"
            f"We received a password reset request for your Fin-TAP account.\n\n"
            f"Click the link below to reset your password "
            f"(valid for 15 minutes):\n\n{reset_url}\n\n"
            f"If you did not request this, please ignore this email.\n\n"
            f"— Fin-TAP Team"
        )
        msg.html = (
            f"<p>Hello <b>{user.name}</b>,</p>"
            f"<p><a href='{reset_url}'>Click here</a> to reset your password (valid for 15 minutes).</p>"
            f"<p>If you did not request this, please ignore this email.</p>"
            f"<p>— Fin-TAP Team</p>"
        )
        mail.send(msg)
        return True
    except Exception as e:
        print(f"[mail] Sending error: {e}")
        return False


@app.route("/forgot-password", methods=["GET", "POST"])
@limiter.limit("5 per hour")
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("root"))

    dev_link = None   # Displayed on the page if mail server is missing

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        user  = User.query.filter_by(email=email).first()

        if user:
            token     = _reset_serializer().dumps(user.id)
            reset_url = url_for("reset_password", token=token, _external=True)
            sent      = _send_reset_email(user, reset_url)
            if not sent:
                # Dev mode — show link directly
                dev_link = reset_url
                print(f"[reset] DEV MODE — reset link: {reset_url}")

        # Prevent email enumeration: same message in all cases
        flash("If you entered a registered email, a reset link has been sent.", "success")

    return render_template("forgot_password.html", dev_link=dev_link)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
@limiter.limit("10 per hour")
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for("root"))

    try:
        user_id = _reset_serializer().loads(token, max_age=900)  # 15 minutes
    except SignatureExpired:
        flash("The password reset link has expired. Please request a new one.", "error")
        return redirect(url_for("forgot_password"))
    except BadSignature:
        flash("Invalid reset link.", "error")
        return redirect(url_for("forgot_password"))

    user = db.session.get(User, user_id)
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password") or ""
        confirm  = request.form.get("confirm")  or ""

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("reset_password.html", token=token)

        ok, msg = _password_strong(password)
        if not ok:
            flash(msg, "error")
            return render_template("reset_password.html", token=token)

        user.password = generate_password_hash(password)
        db.session.commit()
        print(f"[reset] user={user.email} reset their password")
        flash("Your password has been successfully updated. You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html", token=token)


# ── Developer REST API ─────────────────────────────────────────────────────

MAX_KEYS_PER_USER = 5
API_DAILY_LIMIT   = 100   # free tier daily request limit


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _resolve_api_key() -> ApiKey | None:
    """Returns the ApiKey record from the Authorization: Bearer <key> header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    raw = auth[7:].strip()
    if not raw.startswith("fintap_sk_"):
        return None
    h = _hash_key(raw)
    key_obj = ApiKey.query.filter_by(key_hash=h, is_active=True).first()
    return key_obj


def _api_auth_required(f):
    """Decorator: Bearer token validation + daily limit check."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        key_obj = _resolve_api_key()
        if key_obj is None:
            return jsonify({"error": "Invalid or missing API key.",
                            "hint": "Set 'Authorization: Bearer fintap_sk_...' header."}), 401
        if key_obj.requests_today >= API_DAILY_LIMIT:
            return jsonify({"error": f"Daily limit of {API_DAILY_LIMIT} requests reached."}), 429
        # Increment counter
        key_obj.requests_today += 1
        key_obj.last_used_at   = _dt.utcnow()
        db.session.commit()
        # Pass user info to subsequent handler
        request.api_user_id = key_obj.user_id
        return f(*args, **kwargs)
    return decorated


# ── Key Management Page + CRUD ──────────────────────────────────────────────

@app.route("/backtest")
@login_required
def backtest_page():
    w = get_wallet()
    return render_template("backtest.html",
                           trained_stocks=TICKERS_TO_TRAIN,
                           balance=w.balance,
                           user=current_user)


@app.route("/api/backtest/run", methods=["POST"])
@login_required
@limiter.limit("20 per minute")
@csrf.exempt
def api_backtest_run():
    data = request.get_json(silent=True) or {}

    ticker = (data.get("ticker") or "").strip().upper()
    try:
        horizon = int(data.get("horizon", 14))
        lookback = int(data.get("lookback_days", 365))
        capital = float(data.get("start_capital", 1000.0))
    except (TypeError, ValueError):
        return jsonify({"error": "Backtest inputs are invalid. Check horizon, period, and capital."}), 400
    allow_short = bool(data.get("allow_short", False))

    if not ticker or len(ticker) > 10:
        return jsonify({"error": "Invalid ticker."}), 400
    if ticker not in VALID_TICKERS:
        return jsonify({"error": f"{ticker} is not in the tracked instrument list."}), 400
    if horizon not in (7, 14, 30, 90):
        return jsonify({"error": "horizon must be 7, 14, 30 or 90."}), 400
    if lookback not in (90, 180, 365, 730):
        return jsonify({"error": "Invalid lookback_days."}), 400
    if not (100 <= capital <= 1_000_000):
        return jsonify({"error": "start_capital must be between 100 and 1,000,000."}), 400

    try:
        result = run_backtest(
            ticker=ticker,
            horizon=horizon,
            lookback_days=lookback,
            start_capital=capital,
            allow_short=allow_short,
        )
    except Exception as e:
        print(f"[backtest] {ticker}: {e}")
        traceback.print_exc()
        return jsonify({"error": "Backtest engine error. Try refreshing market data or selecting another ticker."}), 500

    if result is None:
        return jsonify({"error": "Insufficient market data for this ticker. Try another ticker or refresh the data cache."}), 422

    return jsonify(result)


@app.route("/developer")
@login_required
def developer_page():
    w = get_wallet()
    return render_template("developer.html",
                           balance=w.balance,
                           user=current_user,
                           daily_limit=API_DAILY_LIMIT)


@app.route("/api/developer/keys", methods=["GET"])
@login_required
@csrf.exempt
def api_dev_keys_list():
    keys = ApiKey.query.filter_by(user_id=current_user.id).order_by(ApiKey.created_at.desc()).all()
    return jsonify([{
        "id":             k.id,
        "name":           k.name,
        "key_prefix":     k.key_prefix,
        "is_active":      k.is_active,
        "requests_today": k.requests_today,
        "last_used_at":   k.last_used_at.strftime("%Y-%m-%d %H:%M") if k.last_used_at else None,
        "created_at":     k.created_at.strftime("%Y-%m-%d"),
    } for k in keys])


@app.route("/api/developer/keys", methods=["POST"])
@login_required
@limiter.limit("10 per hour")
@csrf.exempt
def api_dev_keys_create():
    payload = request.get_json(silent=True) or {}
    name    = str(payload.get("name", "")).strip()[:80]
    if not name:
        return jsonify({"error": "Key name is required."}), 400

    count = ApiKey.query.filter_by(user_id=current_user.id, is_active=True).count()
    if count >= MAX_KEYS_PER_USER:
        return jsonify({"error": f"Maximum {MAX_KEYS_PER_USER} active keys allowed."}), 400

    raw_key = "fintap_sk_" + secrets.token_urlsafe(32)
    key_obj = ApiKey(
        user_id    = current_user.id,
        name       = name,
        key_prefix = raw_key[:18] + "…",
        key_hash   = _hash_key(raw_key),
    )
    db.session.add(key_obj)
    db.session.commit()

    # Return raw key ONLY here — not saved in DB
    return jsonify({"status": "created", "id": key_obj.id,
                    "key": raw_key,   # show and forget
                    "prefix": key_obj.key_prefix}), 201


@app.route("/api/developer/keys/<int:kid>", methods=["DELETE"])
@login_required
@limiter.limit("20 per hour")
@csrf.exempt
def api_dev_keys_revoke(kid):
    key_obj = ApiKey.query.filter_by(id=kid, user_id=current_user.id).first_or_404()
    key_obj.is_active = False
    db.session.commit()
    return jsonify({"status": "revoked"})


# ── General Purpose API v1 ───────────────────────────────────────────────────
# All endpoints require: Authorization: Bearer fintap_sk_...
# Rate limit: 100 daily requests inside @_api_auth_required

@app.route("/api/v1/price/<ticker>")
@csrf.exempt
@limiter.limit("120 per minute")
@_api_auth_required
def apiv1_price(ticker):
    """GET /api/v1/price/<ticker> — current close price."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Unknown ticker."}), 404
    try:
        df = get_processed_data(ticker)
        if df is None or df.empty:
            return jsonify({"error": "No data available."}), 422
        price  = round(float(df["Close"].iloc[-1]), 4)
        prev   = round(float(df["Close"].iloc[-2]), 4)
        change = round((price - prev) / prev * 100, 4)
        return jsonify({
            "ticker":    ticker,
            "price":     price,
            "prev_close": prev,
            "change_pct": change,
            "date":      df.index[-1].strftime("%Y-%m-%d"),
        })
    except Exception:
        return jsonify({"error": "Data unavailable."}), 500


@app.route("/api/v1/ohlc/<ticker>")
@csrf.exempt
@limiter.limit("60 per minute")
@_api_auth_required
def apiv1_ohlc(ticker):
    """GET /api/v1/ohlc/<ticker>?days=30 — OHLCV history."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Unknown ticker."}), 404
    days = min(int(request.args.get("days", 30)), 365)
    try:
        df = get_processed_data(ticker)
        if df is None or df.empty:
            return jsonify({"error": "No data available."}), 422
        df = df.tail(days).dropna(subset=["Open","High","Low","Close"])
        rows = [{"date":  d.strftime("%Y-%m-%d"),
                 "open":  round(float(r["Open"]),  4),
                 "high":  round(float(r["High"]),  4),
                 "low":   round(float(r["Low"]),   4),
                 "close": round(float(r["Close"]), 4),
                 "volume": int(r["Volume"]) if "Volume" in r else None}
                for d, r in df.iterrows()]
        return jsonify({"ticker": ticker, "days": len(rows), "data": rows})
    except Exception:
        return jsonify({"error": "Data unavailable."}), 500


@app.route("/api/v1/sentiment/<ticker>")
@csrf.exempt
@limiter.limit("30 per minute")
@_api_auth_required
def apiv1_sentiment(ticker):
    """GET /api/v1/sentiment/<ticker> — news sentiment analysis."""
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Unknown ticker."}), 404
    try:
        import feedparser
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        feed_ticker = ticker.replace("-USD", "")
        url  = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                f"?s={feed_ticker}&region=US&lang=en-US")
        feed = feedparser.parse(url)
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        articles = []
        for entry in feed.entries[:6]:
            title = (entry.get("title") or "")[:140]
            c = analyzer.polarity_scores(title)["compound"]
            scores.append(c)
            articles.append({"title": title, "score": round(c, 3),
                             "sentiment": "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")})
        avg = round(sum(scores) / len(scores), 3) if scores else 0
        return jsonify({
            "ticker":   ticker,
            "overall":  "bullish" if avg >= 0.05 else ("bearish" if avg <= -0.05 else "neutral"),
            "score":    avg,
            "articles": articles,
        })
    except ImportError:
        return jsonify({"error": "Sentiment packages not installed."}), 503
    except Exception:
        return jsonify({"error": "Sentiment unavailable."}), 500


@app.route("/api/v1/predict/<ticker>")
@csrf.exempt
@limiter.limit("10 per minute")
@_api_auth_required
def apiv1_predict(ticker):
    """
    GET /api/v1/predict/<ticker>?model=LINEAR&horizon=14
    Returns ML prediction spending 1 token.
    """
    ticker  = ticker.upper()
    model   = request.args.get("model", "LINEAR").upper()
    horizon = int(request.args.get("horizon", 14))

    if ticker not in VALID_TICKERS:
        return jsonify({"error": "Unknown ticker."}), 404
    if horizon not in (7, 14, 30, 90):
        return jsonify({"error": "horizon must be 7, 14, 30 or 90."}), 400

    user_id = request.api_user_id
    user    = db.session.get(User, user_id)
    wallet  = Wallet.query.filter_by(user_id=user_id).first()
    if not wallet or wallet.balance < 1:
        return jsonify({"error": "Insufficient token balance."}), 402

    try:
        result, _ = train_and_predict_dynamic(
            ticker, model, list(VALID_FEATURE_GROUPS), horizon=horizon
        )
        if result is None:
            return jsonify({"error": "Prediction failed."}), 422

        wallet.balance -= 1
        pred_rec = Prediction(user_id=user_id, symbol=ticker,
                              model_type=model,
                              predicted_result=str(result.get("prediction", "")))
        db.session.add(pred_rec)
        db.session.commit()

        return jsonify({
            "ticker":          ticker,
            "model":           model,
            "horizon_days":    horizon,
            "prediction":      result.get("prediction"),
            "current_price":   result.get("current_price"),
            "tokens_remaining": wallet.balance,
        })
    except Exception as e:
        print(f"[apiv1_predict] {e}")
        return jsonify({"error": "Prediction engine error."}), 500


# ── Error Handlers ──────────────────────────────────────────────────────
@app.errorhandler(404)
def e404(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found"}), 404
    return redirect(url_for("root"))

@app.errorhandler(429)
def rate_limit_exceeded(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Too many requests. Please wait."}), 429
    flash("Too many requests sent. Please wait a moment.", "error")
    return redirect(url_for("root"))

@app.errorhandler(500)
def e500(e):
    print(f"[500 handler] {e}"); traceback.print_exc()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Server error — check logs"}), 500
    return "<h3>Server error. Wait a few seconds and try again.</h3>", 500

@app.errorhandler(Exception)
def unhandled(e):
    print(f"[unhandled exception] {e}"); traceback.print_exc()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Server error. Please try again."}), 500
    return "<h3>Unexpected error. Please try again.</h3>", 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)