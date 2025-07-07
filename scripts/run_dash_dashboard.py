#!/usr/bin/env python3
"""
Launcher script for the Advanced Electric Vehicle Analysis Dashboard (Dash)
"""

import subprocess
import sys
import os
import webbrowser
import time

def main():
    print("🚗 Starting Advanced Electric Vehicle Analysis Dashboard (Dash)...")
    print("🤖 Loading machine learning models and analytics...")
    print("🌐 Dashboard will open in your browser at http://localhost:8050")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    try:
        # Check if required packages are installed
        print("🔍 Checking dependencies...")
        try:
            import dash
            import plotly
            import sklearn
            print("✅ All dependencies are available")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("💡 Install dependencies with:")
            print("   pip install -r dash_requirements.txt")
            return
        
        # Run the Dash dashboard
        print("🚀 Launching dashboard...")
        subprocess.run([
            sys.executable, "src/dashboards/ev_dash_dashboard.py"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("💡 Make sure you have installed the requirements:")
        print("   pip install -r dash_requirements.txt")

if __name__ == "__main__":
    main() 