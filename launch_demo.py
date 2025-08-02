#!/usr/bin/env python3
"""
🚀 Agentic RAG Demo Launcher
===========================

Easy launcher for the Agentic RAG Educational Demo.
This script handles setup, dependency checking, and launching the demo.

Usage:
    python launch_demo.py

Features:
- ✅ Automatic dependency installation
- ✅ Environment validation
- ✅ Clean demo startup
- ✅ Error handling and guidance
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Display welcome banner"""
    banner = """
    🤖 AGENTIC RAG EDUCATIONAL DEMO
    ================================
    
    🎯 Learn the evolution from Traditional RAG to Intelligent Agentic RAG
    📚 Complete with code walkthrough and interactive examples
    🚀 Ready to run in minutes!
    
    """
    print(banner)

def check_python_version():
    """Ensure compatible Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        print("💡 Please upgrade Python to continue.")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False

def check_demo_files():
    """Verify required demo files exist"""
    required_files = [
        "agentic_rag_demo.py",
        "enhanced_streamlit_app.py", 
        "DEMO.md",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All demo files present")
    return True

def validate_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 API Key Configuration:")
    
    # Check for hardcoded keys in demo file
    demo_file = Path("agentic_rag_demo.py")
    if demo_file.exists():
        content = demo_file.read_text()
        if "gsk_" in content and "b727392ea5c4e76905ada9e07f7646ba4fc58e47" in content:
            print("✅ Demo API keys found (hardcoded for demo)")
            return True
    
    # Check environment variables
    groq_key = os.getenv("GROQ_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    
    if groq_key and serper_key:
        print("✅ API keys found in environment variables")
        return True
    
    print("⚠️  API keys not found in environment")
    print("💡 Demo will use hardcoded keys for educational purposes")
    return True

def choose_demo_version():
    """Let user choose which demo to run"""
    print("\n🎮 Choose Demo Version:")
    print("1. 📚 Educational Demo (agentic_rag_demo.py) - Heavily commented, perfect for learning")
    print("2. 🚀 Enhanced Demo (enhanced_streamlit_app.py) - Full-featured with PDF upload")
    
    while True:
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
        if choice == "" or choice == "1":
            return "agentic_rag_demo.py"
        elif choice == "2":
            return "enhanced_streamlit_app.py"
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

def launch_streamlit(demo_file):
    """Launch the Streamlit demo"""
    print(f"\n🚀 Launching {demo_file}...")
    print("✨ Features available:")
    print("   📊 Side-by-side Traditional vs Agentic RAG comparison")
    print("   🧭 Intelligent routing with detailed explanations")
    print("   🔍 Complete source attribution and transparency")
    print("   📈 Interactive visualizations and metrics")
    print("   💬 Sample questions for different scenarios")
    print()
    print("🔗 Demo will open at: http://localhost:8501")
    print("⏹️  Press Ctrl+C in terminal to stop")
    print("-" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", demo_file,
            "--server.headless", "true",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching demo: {e}")
        print("💡 Try running manually: streamlit run", demo_file)

def show_next_steps():
    """Display helpful next steps"""
    print("\n📚 What to do next:")
    print("1. 🔍 Read DEMO.md for comprehensive explanation")
    print("2. 💻 Explore the commented code in agentic_rag_demo.py")
    print("3. 🧪 Try different query types to see routing decisions")
    print("4. 🛠️ Modify the code to experiment with your own data")
    print("5. 🚀 Build your own Agentic RAG system!")
    print()
    print("🎯 Key files to explore:")
    print("   📄 DEMO.md - Complete educational guide")
    print("   🐍 agentic_rag_demo.py - Heavily commented implementation")
    print("   🚀 enhanced_streamlit_app.py - Full-featured version")
    print("   📦 requirements.txt - All dependencies")

def main():
    """Main launcher function"""
    print_banner()
    
    # Pre-flight checks
    if not check_python_version():
        return
    
    if not check_demo_files():
        return
    
    # Install dependencies
    print("🔍 Checking dependencies...")
    try:
        import streamlit
        print("✅ Dependencies already installed")
    except ImportError:
        if not install_requirements():
            return
    
    # Validate API keys
    validate_api_keys()
    
    # Choose demo version
    demo_file = choose_demo_version()
    
    # Launch demo
    launch_streamlit(demo_file)
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()