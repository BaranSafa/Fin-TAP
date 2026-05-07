"""
run.py  —  Fin-TAP Geliştirme Sunucusu Başlatıcı
==================================================
Bu dosyayı çalıştırmak yeterli: `python run.py`
Sunucu başladıktan sonra tarayıcından http://127.0.0.1:5000 adresine git.

NOT: Bu dosya sadece YEREL GELİŞTİRME içindir.
     Render / Heroku gibi sunucularda gunicorn kullanılır (Procfile'da tanımlı).
"""
from app import app
import os
import sys

# Windows'ta terminal bazen donabilir — konsol modunu düzelterek bunu önleriz
if sys.platform.startswith('win'):
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

if __name__ == '__main__':
    print("----------------------------------------------------------------")
    print(">>> Fin-TAP Web Sunucusu Başlatılıyor...")
    print(">>> Tarayıcını aç ve şu adrese git: http://127.0.0.1:5000")
    print("----------------------------------------------------------------")

    # debug=True  → kodda değişiklik yaparsan sunucu otomatik yenilenir
    # host='0.0.0.0' → yalnızca localhost değil, aynı WiFi'deki cihazlar da erişebilir
    #                  (örn. telefonundan test etmek için bilgisayarının yerel IP'sini kullanabilirsin)
    app.run(host='0.0.0.0', port=5000, debug=True)