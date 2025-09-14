import os
import sys
import subprocess
import time
import webbrowser
import signal
from threading import Thread

def run_uvicorn():
    subprocess.run([
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--reload",
        "--port",
        "8000"
    ])

def run_frontend():
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    subprocess.run(["npm", "run", "dev"], cwd=frontend_dir)

def open_browser():
    time.sleep(5)
    webbrowser.open("http://localhost:8080/")

def orchestrate():
    threads = []
    t1 = Thread(target=run_uvicorn, daemon=True)
    t2 = Thread(target=run_frontend, daemon=True)
    threads.append(t1)
    threads.append(t2)
    t1.start()
    t2.start()
    open_browser()
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down servers...")
        os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    print("Starting backend, frontend, and UI...")
    orchestrate()
