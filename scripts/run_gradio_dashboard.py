#!/usr/bin/env python3
"""
Launcher script for the AI-Powered Electric Vehicle Analysis Dashboard (Gradio)
"""

import subprocess
import sys
import os
import webbrowser
import time

def main():
    print("🚗 Starting AI-Powered Electric Vehicle Analysis Dashboard (Gradio)...")
    print("🤖 Loading AI models and machine learning insights...")
    print("🌐 Dashboard will open in your browser at http://localhost:7861")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 70)
    
    try:
        # Check if required packages are installed
        print("🔍 Checking dependencies...")
        try:
            import gradio
            import plotly
            import sklearn
            print("✅ All dependencies are available")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("💡 Install dependencies with:")
            print("   pip install -r gradio_requirements.txt")
            return
        
        # Run the Gradio dashboard
        print("🚀 Launching AI dashboard...")
        subprocess.run([
            sys.executable, "src/dashboards/ev_gradio_dashboard.py"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("💡 Make sure you have installed the requirements:")
        print("   pip install -r gradio_requirements.txt")

if __name__ == "__main__":
    main() 