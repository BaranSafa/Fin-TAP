# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

# 1. KULLANICI TABLOSU
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Kullanıcı ile Cüzdanı bağlıyoruz (1-to-1)
    wallet = db.relationship('Wallet', backref='user', uselist=False)

# 2. CÜZDAN (WALLET) TABLOSU - TOKEN SİSTEMİ İÇİN
class Wallet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True)
    balance = db.Column(db.Integer, default=5)  # Kayıt olana 5 Token hediye!
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# 3. İŞLEM GEÇMİŞİ (TRANSACTION)
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    amount_paid = db.Column(db.Float, nullable=False)
    tokens_added = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)