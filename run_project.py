#!/usr/bin/env python3
"""
Main launcher for Electric Vehicle Analysis Project
Choose which dashboard to run
"""

import subprocess
import sys
import os

def main():
    print("🚗 Electric Vehicle Analysis Project")
    print("=" * 50)
    print("Choose a dashboard to run:")
    print("1. Streamlit Dashboard (Interactive & User-friendly)")
    print("2. Dash Dashboard (Advanced Analytics & ML)")
    print("3. Gradio Dashboard (AI-Powered & Modern UI)")
    print("4. Exit")
    print("-" * 50)
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting Streamlit Dashboard...")
                subprocess.run([
                    sys.executable, "scripts/run_dashboard.py"
                ])
                break
            elif choice == "2":
                print("\n🚀 Starting Dash Dashboard...")
                subprocess.run([
                    sys.executable, "scripts/run_dash_dashboard.py"
                ])
                break
            elif choice == "3":
                print("\n🚀 Starting Gradio Dashboard...")
                subprocess.run([
                    sys.executable, "scripts/run_gradio_dashboard.py"
                ])
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 