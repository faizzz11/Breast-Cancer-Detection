# 🚀 Deployment Guide

This guide will help you deploy your Breast Cancer Detection app to various platforms.

## 📋 **Pre-Deployment Checklist**

- [ ] All code is working locally
- [ ] Model is trained (`models/final_model.joblib` exists)
- [ ] Requirements.txt is up to date
- [ ] README.md is complete
- [ ] Code is pushed to GitHub

## 🌟 **Option 1: Streamlit Community Cloud (Recommended)**

**Pros:** Free, easy, official Streamlit hosting
**Cons:** Limited resources

### Steps:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Repository: `yourusername/breast-cancer-detection`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - Click "Deploy!"

3. **Your app will be live at:**
   `https://yourusername-breast-cancer-detection-app-streamlit-app-xyz.streamlit.app`

## 🤗 **Option 2: Hugging Face Spaces**

**Pros:** Great for ML projects, good visibility, free
**Cons:** Slightly more complex setup

### Steps:

1. **Create Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Name: `breast-cancer-detection`
   - SDK: `Streamlit`
   - License: `MIT`

2. **Upload Files:**
   - Upload all project files
   - Or connect your GitHub repository

3. **Configure:**
   - Ensure `requirements.txt` is in root directory
   - Main app file should be `app.py` (rename if needed)

4. **Your app will be live at:**
   `https://huggingface.co/spaces/yourusername/breast-cancer-detection`

## 🚂 **Option 3: Railway**

**Pros:** More powerful, supports databases, good for complex apps
**Cons:** Limited free tier

### Steps:

1. **Connect Repository:**
   - Go to [railway.app](https://railway.app)
   - Sign up/Login with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure:**
   - Railway auto-detects Python
   - Add environment variables if needed
   - Deploy automatically

3. **Custom Domain:**
   - Railway provides a custom domain
   - You can also connect your own domain

## ☁️ **Option 4: Heroku**

**Pros:** Popular, reliable, many add-ons
**Cons:** No longer has free tier

### Steps:

1. **Install Heroku CLI:**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Windows
   # Download from heroku.com
   ```

2. **Deploy:**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

3. **Configure:**
   - Ensure `Procfile` exists (already created)
   - Set buildpack: `heroku buildpacks:set heroku/python`

## 🔧 **Troubleshooting**

### Common Issues:

1. **"Module not found" errors:**
   - Check `requirements.txt` includes all dependencies
   - Ensure file paths are relative, not absolute

2. **"Model not found" errors:**
   - Ensure `models/final_model.joblib` is included in repository
   - Check file size limits (GitHub: 100MB, Streamlit Cloud: 1GB)

3. **Memory errors:**
   - Optimize model size
   - Use `@st.cache_resource` for model loading
   - Consider model compression

4. **Slow loading:**
   - Add loading spinners
   - Cache data loading with `@st.cache_data`
   - Optimize imports

### File Size Limits:

- **GitHub:** 100MB per file
- **Streamlit Cloud:** 1GB total
- **Hugging Face:** 10GB total
- **Railway:** Depends on plan

### If Model is Too Large:

```python
# Option 1: Use Git LFS
git lfs track "*.joblib"
git add .gitattributes
git add models/final_model.joblib
git commit -m "Add model with LFS"

# Option 2: Download model on startup
# Add to streamlit_app.py:
@st.cache_resource
def download_model():
    # Download from cloud storage
    pass
```

## 📊 **Monitoring Your Deployment**

1. **Check app logs** for errors
2. **Monitor usage** through platform analytics
3. **Set up alerts** for downtime
4. **Regular updates** and maintenance

## 🎉 **Post-Deployment**

1. **Update README** with live app URL
2. **Share on social media** (LinkedIn, Twitter)
3. **Add to portfolio** website
4. **Submit to showcases** (Streamlit gallery, etc.)
5. **Collect feedback** and iterate

## 📞 **Need Help?**

- **Streamlit Community:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **Hugging Face:** [discuss.huggingface.co](https://discuss.huggingface.co)
- **Railway Discord:** [railway.app/discord](https://railway.app/discord)

---

**Good luck with your deployment! 🚀**