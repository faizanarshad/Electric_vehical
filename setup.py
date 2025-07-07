#!/usr/bin/env python3
"""
Setup script for Electric Vehicle Analysis Project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all project requirements"""
    print("🔧 Setting up Electric Vehicle Analysis Project...")
    print("📦 Installing dependencies...")
    
    try:
        # Install main requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All dependencies installed successfully!")
        
        # Install additional dashboard-specific requirements
        print("📊 Installing dashboard-specific requirements...")
        
        # Streamlit requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "config/dashboard_requirements.txt"
        ])
        
        # Dash requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "config/dash_requirements.txt"
        ])
        
        # Gradio requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "config/gradio_requirements.txt"
        ])
        
        print("✅ Setup completed successfully!")
        print("\n🚀 You can now run the project with:")
        print("   python run_project.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("💡 Try running: pip install --upgrade pip")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚗 Electric Vehicle Analysis Project Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Error: requirements.txt not found!")
        print("💡 Make sure you're in the project root directory")
        return
    
    # Install requirements
    if install_requirements():
        print("\n🎉 Project setup completed!")
        print("📚 Check the README.md for usage instructions")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 