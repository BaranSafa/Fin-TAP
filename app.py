# --- app.py GÜNCEL HALİ ---
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, Wallet, Transaction # models.py'den çektik
import os

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
app.config['SECRET_KEY'] = 'bu_cok_gizli_bir_anahtardir_fin_tap_v2'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///fintap.db')
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db.init_app(app) # Veritabanını bağla

# --- LOGIN MANAGER AYARLARI ---
login_manager = LoginManager()
login_manager.login_view = 'login' # Giriş yapmamışsa buraya at
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- VERİTABANI OLUŞTURMA (İLK ÇALIŞTIRMADA) ---
with app.app_context():
    db.create_all()

# ==========================================
# 1. STATE: GUEST STATE (Giriş Yapmamış)
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Hatalı e-posta veya şifre!', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        
        # Kullanıcı var mı kontrol et
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Bu e-posta zaten kayıtlı.', 'error')
            return redirect(url_for('register'))
        
        # Yeni kullanıcı oluştur
        new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))
        db.session.add(new_user)
        db.session.commit()
        
        # Kullanıcıya Cüzdan Oluştur (5 Token Hediye)
        new_wallet = Wallet(user_id=new_user.id, balance=5)
        db.session.add(new_wallet)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('dashboard'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ==========================================
# 2. STATE: AUTHENTICATED STATE (Giriş Yapmış)
# ==========================================

@app.route('/')
@login_required # Sadece giriş yapanlar görebilir
def dashboard():
    # Cüzdan bilgisini al
    wallet = Wallet.query.filter_by(user_id=current_user.id).first()
    return render_template('index.html', user=current_user, balance=wallet.balance)

# ==========================================
# 3. STATE: RESTRICTED & PROCESSING STATE
# ==========================================

@app.route('/api/predict_check/<ticker>')
@login_required
def predict_check(ticker):
    """
    Bu endpoint, kullanıcı 'Analiz Et' butonuna bastığında çalışır.
    Backend'de bakiye kontrolü yapar.
    """
    wallet = Wallet.query.filter_by(user_id=current_user.id).first()
    
    # --- RESTRICTED STATE KONTROLÜ ---
    if wallet.balance <= 0:
        return jsonify({"status": "restricted", "message": "Yetersiz Bakiye! Lütfen Token alın."})
    
    # --- PROCESSING STATE ---
    # Bakiye var, 1 Token düş
    wallet.balance -= 1
    db.session.commit()
    
    # Burada normalde AI fonksiyonunu çağıracağız
    # Şimdilik frontende 'devam et' sinyali veriyoruz
    return jsonify({
        "status": "processing", 
        "new_balance": wallet.balance,
        "message": "Analiz başlatılıyor... (1 Token harcandı)"
    })

# --- Eski fonksiyonların buraya entegre edilecek ---
# (train_and_predict vb. fonksiyonları buraya import etmeyi unutma)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# --- SİSTEM BAŞLATICI ---
def check_system():
    if not os.path.exists("models"): os.makedirs("models")
    tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
    for t in tickers:
        p = os.path.join("models", f"{t}_scaler.joblib")
        if not os.path.exists(p): joblib.dump("dummy", p)

def get_trained_stocks():
    check_system()
    return [f.split('_')[0] for f in os.listdir("models") if f.endswith("_scaler.joblib")]

# --- SAYFA ROTALARI ---
@app.route('/')
def index(): return render_template('home.html', stocks=get_trained_stocks())

@app.route('/predict')
def predict():
    return render_template('predict.html', 
                           ticker_from_url=request.args.get('ticker'), 
                           trained_stocks=get_trained_stocks())

@app.route('/all_stocks')
def all_stocks(): return render_template('all_stocks.html', stocks=get_trained_stocks())

# --- app.py İÇİNDEKİ '/suggestion' KISMINI SİLİP BUNU YAPIŞTIR ---

@app.route('/compare')
def compare():
    # Sayfaya hisse listesini gönderiyoruz ki dropdown'da seçebilsin
    return render_template('compare.html', stocks=get_trained_stocks())

# --- app.py GÜNCELLEMESİ ---

@app.route('/api/compare_stocks')
def api_compare_stocks():
    t1 = request.args.get('ticker1')
    t2 = request.args.get('ticker2')
    
    if not t1 or not t2:
        return jsonify({"error": "İki hisse seçmelisiniz"}), 400

    try:
        results = {}
        for t in [t1, t2]:
            # Modeli çalıştır (Gelecek 14 günü döndürür)
            future_predictions, _ = train_and_predict_dynamic(t, 'LINEAR', ['RSI', 'MACD', 'Volatility'])
            df = get_processed_data(t)
            
            if future_predictions is None or df is None:
                return jsonify({"error": f"{t} verisi veya modeli bulunamadı. Lütfen train.py çalıştırın."}), 500
                
            # --- HATA ÇÖZÜMÜ ---
            # Gelen veri bir liste olduğu için [-1] ile son günün tahminini alıyoruz.
            if isinstance(future_predictions, list):
                future_price = future_predictions[-1] # 14. Günün tahmini
            else:
                future_price = future_predictions # Tek sayı gelirse direkt al
            
            last_price = df['Close'].iloc[-1]
            rsi = df['RSI_14'].iloc[-1]
            volatility = df['volatility'].iloc[-1]
            
            # Getiri Hesabı
            gain = ((future_price / last_price) - 1) * 100
            
            results[t] = {
                "price": round(last_price, 2),
                "predicted": round(future_price, 2), # Artık hata vermez, çünkü float
                "gain": round(gain, 2),
                "rsi": round(rsi, 2),
                "volatility": round(volatility, 3)
            }
            
        return jsonify(results)

    except Exception as e:
        print(f"Compare Hatası: {e}")
        return jsonify({"error": f"Sistem hatası: {str(e)}"}), 500
    


    

@app.route('/prices')
def prices(): return render_template('prices.html')

@app.route('/roadmap')
def roadmap(): return render_template('roadmap.html')

# --- API ROTALARI ---

# 1. TAHMİN API (Model eğitir, yavaştır)
@app.route('/api/get_data/<string:ticker>')
def get_data(ticker):
    mt = request.args.get('model', 'LINEAR')
    ft = request.args.get('features', '').split(',')
    fp, vd = train_and_predict_dynamic(ticker, mt, ft)
    if fp is None: return jsonify({"error": "Tahmin hatası"}), 500
    return jsonify({
        "ticker": ticker, "model": mt, "future_prediction_7day": fp, "validation_data": vd,
        "last_known_date": vd["dates"][-1], "last_known_price": vd["actual_prices"][-1]
    })

# 2. GEÇMİŞ VERİ API (Sadece grafik içindir, ÇOK HIZLIDIR)
@app.route('/api/history/<string:ticker>')
def get_history(ticker):
    df = get_processed_data(ticker)
    if df is None or df.empty: return jsonify({"error": "Veri yok"}), 404
    
    # Son 30 günü al
    df_last = df.iloc[-30:]
    return jsonify({
        "dates": [d.strftime('%Y-%m-%d') for d in df_last.index],
        "prices": df_last['Close'].tolist()
    })

@app.route('/api/market_summary')
def market_summary():
    summary = []
    tickers = get_trained_stocks()
    
    for ticker in tickers:
        try:
            # Sadece son 2 günün verisini çek (Hız için)
            df = get_processed_data(ticker)
            if df is not None and len(df) >= 2:
                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                change_pct = ((last_price - prev_price) / prev_price) * 100
                
                summary.append({
                    "ticker": ticker,
                    "price": round(last_price, 2),
                    "change": round(change_pct, 2),
                    "trend": "up" if change_pct > 0 else "down"
                })
        except:
            continue
            
    return jsonify(summary)

@app.route('/db-kur')
def db_kur():
    try:
        with app.app_context():
            db.create_all()
        return "<h1>BAŞARILI! Tablolar oluşturuldu. Şimdi /register sayfasına gidip kayıt olabilirsin.</h1>"
    except Exception as e:
        return f"<h1>HATA OLUŞTU: {str(e)}</h1>"


if __name__ == '__main__':
    check_system()
    app.run(debug=True, port=5000)  