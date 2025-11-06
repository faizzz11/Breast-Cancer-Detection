#!/usr/bin/env python3
"""
Deployment preparation script for the Breast Cancer Detection project.
This script checks if everything is ready for deployment.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_file_size(filepath, max_size_mb=100):
    """Check if file size is within limits."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb <= max_size_mb:
            print(f"✅ File size OK: {filepath} ({size_mb:.1f}MB)")
            return True
        else:
            print(f"⚠️  Large file: {filepath} ({size_mb:.1f}MB) - Consider Git LFS")
            return False
    return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported."""
    try:
        sys.path.append('src')
        # Try importing key modules
        from data_loader import load_data
        import joblib
        
        # Test model loading
        if os.path.exists('models/final_model.joblib'):
            model = joblib.load('models/final_model.joblib')
            print("✅ Model loads successfully")
        
        # Test data loading
        X, y = load_data(source='sklearn')
        print(f"✅ Data loads successfully: {X.shape}")
        
        return True
    except Exception as e:
        print(f"❌ App test failed: {e}")
        return False

def check_git_status():
    """Check git repository status."""
    try:
        # Check if git repo exists
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository initialized")
            
            # Check for uncommitted changes
            if "nothing to commit" in result.stdout:
                print("✅ No uncommitted changes")
            else:
                print("⚠️  Uncommitted changes detected")
                print("   Run: git add . && git commit -m 'Ready for deployment'")
            
            return True
        else:
            print("❌ Not a git repository")
            print("   Run: git init && git add . && git commit -m 'Initial commit'")
            return False
    except FileNotFoundError:
        print("❌ Git not installed")
        return False

def main():
    """Main deployment preparation check."""
    
    print("🚀 Deployment Preparation Checklist")
    print("=" * 50)
    
    all_good = True
    
    # Check essential files
    print("\n📁 Essential Files:")
    essential_files = [
        ("app/streamlit_app.py", "Streamlit app"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation"),
        (".gitignore", "Git ignore file"),
        ("models/final_model.joblib", "Trained model")
    ]
    
    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check file sizes
    print("\n📏 File Sizes:")
    large_files = ["models/final_model.joblib"]
    for filepath in large_files:
        if os.path.exists(filepath):
            check_file_size(filepath)
    
    # Test app functionality
    print("\n🧪 App Testing:")
    if not test_streamlit_app():
        all_good = False
    
    # Check git status
    print("\n📝 Git Status:")
    check_git_status()
    
    # Deployment options
    print("\n🌐 Deployment Options:")
    print("1. Streamlit Community Cloud (Recommended)")
    print("   → https://share.streamlit.io")
    print("2. Hugging Face Spaces")
    print("   → https://huggingface.co/spaces")
    print("3. Railway")
    print("   → https://railway.app")
    
    # Final status
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Ready for deployment!")
        print("\nNext steps:")
        print("1. Push to GitHub: git push origin main")
        print("2. Go to your chosen platform")
        print("3. Connect repository and deploy")
        print("4. Update README with live URL")
    else:
        print("⚠️  Please fix the issues above before deploying")
    
    print("\n📖 See DEPLOYMENT.md for detailed instructions")

if __name__ == "__main__":
    main()