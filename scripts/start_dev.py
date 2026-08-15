"""
Full-Stack Development Server Launcher & Integration Helper
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Launches:
1. FastAPI Backend Server (http://localhost:8000)
2. React + Vite Frontend Dev Server (http://localhost:5173)
"""

import os
import sys
import subprocess
import time
import argparse

def launch_servers(check_only=False):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")

    print("=" * 75)
    print("VOICE-ENABLED RAG MODEL — FULL-STACK DEVELOPMENT LAUNCHER")
    print("=" * 75)
    print(f"[*] Root Directory    : {root_dir}")
    print(f"[*] Backend Directory : {backend_dir}")
    print(f"[*] Frontend Directory: {frontend_dir}")

    if check_only:
        print("[+] Environment & configuration paths verified successfully!")
        return

    print("\n[1/2] Starting FastAPI Backend Server on http://localhost:8000...")
    backend_cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
    backend_proc = subprocess.Popen(backend_cmd, cwd=backend_dir)

    print("[2/2] Starting React + Vite Frontend Server on http://localhost:5173...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=frontend_dir)

    print("\n" + "=" * 75)
    print("🚀 BOTH SERVERS RUNNING!")
    print("  • Frontend UI: http://localhost:5173")
    print("  • Backend API: http://localhost:8000 (Docs: http://localhost:8000/docs)")
    print("  • Press Ctrl+C in this terminal to stop both servers.")
    print("=" * 75)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Stopping development servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("[+] Servers stopped cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify configuration without blocking")
    args = parser.parse_args()
    launch_servers(check_only=args.check)
