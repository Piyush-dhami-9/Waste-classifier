"""
SmartWaste AI - Single Launcher
Starts:
  1. Web Server (classification UI + dashboard) on port 8000
  2. Camera Detection (littering detection) - optional
"""

import subprocess
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def start_web_server():
    """Start the FastAPI server (classification + dashboard)."""
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=os.path.join(PROJECT_ROOT, "backend"),
    )


def start_camera_detection():
    """Start the littering detection camera system."""
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=PROJECT_ROOT,
    )


def main():
    print("=" * 60)
    print("  SmartWaste AI - Unified Launcher")
    print("=" * 60)
    print()
    print("  [1] Web Server only (classification + dashboard)")
    print("  [2] Camera Detection only (littering detection)")
    print("  [3] Both (full system)")
    print()

    choice = input("  Choose mode [1/2/3]: ").strip()
    
    processes = []
    
    if choice in ("1", "3"):
        print("\n  Starting Web Server on http://localhost:8000 ...")
        print("    - Classification UI: http://localhost:8000")
        print("    - Dashboard:         http://localhost:8000/dashboard")
        processes.append(start_web_server())
        time.sleep(2)
    
    if choice in ("2", "3"):
        print("\n  Starting Camera Detection...")
        print("    - Press 'q' in camera window to quit")
        print("    - Press 'r' to reset tracking")
        processes.append(start_camera_detection())
    
    if not processes:
        print("  Invalid choice. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("  System running. Press Ctrl+C to stop all.")
    print("=" * 60)
    
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        print("  Done.")


if __name__ == "__main__":
    main()
