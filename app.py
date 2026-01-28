import os
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

# models.py dosyasından tabloları çekiyoruz
from models import db, User, Wallet, Transaction

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

# --- AYARLAR (CONFIG) ---
app.config['SECRET_KEY'] = 'bu_cok_gizli_bir_anahtardir_fin_tap_v2_enterprise'

# Veritabanı Ayarı: Render'daysa orayı kullan, yoksa yerel dosyayı kullan
database_url = os.environ.get('DATABASE_URL', 'sqlite:///fintap.db')
# Render'ın verdiği postgres:// adresini postgresql:// yap (SQLAlchemy uyumu için)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Eklentileri Başlat
CORS(app)
db.init_app(app)

# --- LOGIN MANAGER (Kullanıcı Giriş Yönetimi) ---
login_manager = LoginManager()
login_manager.login_view = 'login' # Giriş yapmamış biri dashboard'a girmeye çalışırsa buraya at
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ========================================================
# 1. GUEST STATE & YÖNLENDİRME MANTIĞI
# ========================================================

@app.route('/')
def index():
    """
    Ana yönlendirme merkezi.
    - Kullanıcı giriş yapmışsa -> Dashboard'a (Authenticated State)
    - Yapmamışsa -> Landing Page'e (Guest State)
    """
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

# ========================================================
# 2. AUTHENTICATED STATE (Üye Alanı)
# ========================================================

@app.route('/dashboard')
@login_required  # Sadece üyeler girebilir
def dashboard():
    # Kullanıcının cüzdanını bul
    wallet = Wallet.query.filter_by(user_id=current_user.id).first()
    
    # Eğer cüzdan yoksa (hata durumu), hemen oluştur
    if not wallet:
        wallet = Wallet(user_id=current_user.id, balance=5)
        db.session.add(wallet)
        db.session.commit()
    
    # Dashboard sayfasına bakiye bilgisini gönder
    return render_template('dashboard.html', balance=wallet.balance, user=current_user)

# ========================================================
# 3. RESTRICTED & PROCESSING STATE (İşlem Mantığı)
# ========================================================

@app.route('/api/predict_check', methods=['POST'])
@login_required
def predict_check():
    """
    Bu API, kullanıcı 'Analiz Et' butonuna bastığında çağrılır.
    RESTRICTED STATE kontrolü burada yapılır.
    """
    wallet = Wallet.query.filter_by(user_id=current_user.id).first()
    
    # --- KONTROL: RESTRICTED STATE ---
    if wallet.balance <= 0:
        return jsonify({
            "status": "restricted",
            "message": "Yetersiz Bakiye! Analiz yapmak için Token satın almalısınız."
        }), 402  # 402 Payment Required kodu

    # --- İŞLEM: PROCESSING STATE ---
    # Bakiyesi var, 1 Token düş
    wallet.balance -= 1
    
    # (Opsiyonel) İşlem geçmişine kaydet
    # new_tx = Transaction(user_id=current_user.id, amount_paid=0, tokens_added=-1)
    # db.session.add(new_tx)
    
    db.session.commit()
    
    return jsonify({
        "status": "success",
        "new_balance": wallet.balance,
        "message": "Analiz başlatılıyor... (1 Token kullanıldı)"
    })

# ========================================================
# 4. KAYIT VE GİRİŞ İŞLEMLERİ
# ========================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Zaten giriş yapmışsa Dashboard'a yolla
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        
        # 1. Kontrol: Bu mail zaten var mı?
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Bu e-posta adresi zaten kayıtlı. Lütfen giriş yapın.', 'error')
            return redirect(url_for('register'))
        
        # 2. Kayıt: Yeni kullanıcı oluştur
        new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))
        db.session.add(new_user)
        db.session.commit()
        
        # 3. Ödül: Cüzdan oluştur ve 5 Token ver
        new_wallet = Wallet(user_id=new_user.id, balance=5)
        db.session.add(new_wallet)
        db.session.commit()
        
        # 4. Giriş: Otomatik giriş yap ve panele at
        login_user(new_user)
        flash('Aramıza hoş geldin! 5 Hediye Token hesabına tanımlandı.', 'success')
        return redirect(url_for('dashboard'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Hatalı e-posta veya şifre girdiniz.', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index')) # Çıkınca Landing Page'e at

# ========================================================
# 5. YARDIMCI ARAÇLAR (Veritabanı Kurulumu)
# ========================================================

@app.route('/db-kur')
def db_kur():
    """
    Render üzerinde tabloları elle kurmak için 'Arka Kapı' linki.
    /db-kur adresine gidince çalışır.
    """
    try:
        with app.app_context():
            db.create_all()
        return "<h1 style='color:green'>BAŞARILI! Tablolar (User, Wallet) oluşturuldu.</h1>"
    except Exception as e:
        return f"<h1 style='color:red'>HATA: {str(e)}</h1>"

# Uygulama Başlangıcı
if __name__ == '__main__':
    # Lokal çalışırken de tabloları kontrol et
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)