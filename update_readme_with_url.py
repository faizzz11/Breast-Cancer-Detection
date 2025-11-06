#!/usr/bin/env python3
"""
Script to update README.md with your deployed app URL.
Run this after deploying to update the badges and links.
"""

def update_readme_with_url(app_url):
    """Update README.md with the deployed app URL."""
    
    # Read current README
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Replace placeholder URLs
    content = content.replace('https://your-app-url-here.streamlit.app', app_url)
    
    # Write updated README
    with open('README.md', 'w') as f:
        f.write(content)
    
    print(f"✅ README.md updated with URL: {app_url}")

def main():
    """Main function to get URL and update README."""
    print("🔗 README URL Updater")
    print("=" * 30)
    
    print("\nAfter deploying your app, enter the live URL here:")
    print("Examples:")
    print("- Streamlit Cloud: https://username-breast-cancer-detection-app-streamlit-app-abc123.streamlit.app")
    print("- Hugging Face: https://huggingface.co/spaces/username/breast-cancer-detection")
    print("- Railway: https://breast-cancer-detection-production.up.railway.app")
    
    app_url = input("\nEnter your deployed app URL: ").strip()
    
    if app_url and app_url.startswith('http'):
        update_readme_with_url(app_url)
        print("\n🎉 README updated successfully!")
        print("Don't forget to commit and push the changes:")
        print("git add README.md && git commit -m 'Update README with live app URL' && git push")
    else:
        print("❌ Invalid URL. Please enter a valid HTTP/HTTPS URL.")

if __name__ == "__main__":
    main()