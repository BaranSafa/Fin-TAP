import webview
from app import app
import sys
import threading

def start_server():
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':

    print("Starting desktop application...")

    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()
    
    print("The server was started in the background (http://127.0.0.1:5000).")

    window = webview.create_window(
        'Fin-TAP: Financial Forecasting Platform', 
        'http://127.0.0.1:5000/',           
        width=1000,                         
        height=750,                         
        resizable=True                      
    )
    
    webview.start(debug=False) 
    
    print("The application has been closed.")
    sys.exit()