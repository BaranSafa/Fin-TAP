from app import app
import os
import sys

# Windows'ta bazen print komutları donmaya sebep olabilir, bunu engellemek için:
if sys.platform.startswith('win'):
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

if __name__ == '__main__':
    print("----------------------------------------------------------------")
    print(">>> Fin-TAP Web Sunucusu Başlatılıyor...")
    print(">>> Tarayıcını aç ve şu adrese git: http://127.0.0.1:5000")
    print("----------------------------------------------------------------")
    
    # debug=True: Kodda değişiklik yaparsan sunucu otomatik yenilenir (Geliştirme için harika)
    # host='0.0.0.0': Sadece senin bilgisayarından değil, aynı ağdaki telefondan bile girilebilir
    app.run(host='0.0.0.0', port=5000, debug=True)