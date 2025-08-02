#!/usr/bin/env python3
"""
Enhanced Agentic RAG UI Launcher

Launch the enhanced web interface with PDF upload and dynamic training capabilities.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for the enhanced UI"""
    print("📦 Installing enhanced UI dependencies...")
    requirements = [
        "streamlit>=1.28.0",
        "plotly>=5.17.0", 
        "pandas>=1.5.0",
        "langchain-groq>=0.1.0",
        "faiss-cpu>=1.7.4",
        "langchain>=0.1.0",
        "langchain-community>=0.1.0",
        "langchain-huggingface>=0.1.0",
        "sentence-transformers>=2.3.0",
        "pypdf2>=3.0.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0"
    ]
    
    try:
        for req in requirements:
            print(f"Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_streamlit():
    """Check if streamlit is available"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def launch_enhanced_ui():
    """Launch the enhanced Streamlit application"""
    print("🚀 Launching Enhanced Agentic RAG Web Interface...")
    print("✨ New Features:")
    print("   📄 PDF Upload & Training")
    print("   🧠 Dynamic Question Generation") 
    print("   🎯 Better Traditional vs Agentic RAG Differentiation")
    print("   📊 Enhanced Visual Pipelines")
    print("   🔍 Advanced Web Search Integration")
    print()
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "enhanced_streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Enhanced Agentic RAG UI...")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def main():
    print("🤖 Enhanced Agentic RAG Web Interface Launcher")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("enhanced_streamlit_app.py"):
        print("❌ enhanced_streamlit_app.py not found!")
        print("💡 Please run this script from the agentic-rag directory")
        return
    
    # Check if streamlit is installed
    if not check_streamlit():
        print("📦 Streamlit not found, installing dependencies...")
        if not install_requirements():
            print("❌ Failed to install dependencies")
            return
    else:
        print("✅ Streamlit found!")
    
    print()
    launch_enhanced_ui()

if __name__ == "__main__":
    main()