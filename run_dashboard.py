#!/usr/bin/env python3
"""
Launcher script for the Electric Vehicle Analysis Dashboard
"""

import subprocess
import sys
import os

def main():
    print("🚗 Starting Electric Vehicle Analysis Dashboard...")
    print("📊 Loading data and launching Streamlit...")
    print("🌐 Dashboard will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run the Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "ev_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("💡 Make sure you have installed the requirements:")
        print("   pip install -r dashboard_requirements.txt")

if __name__ == "__main__":
    main() 