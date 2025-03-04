# Deploying to Streamlit Cloud

This guide will help you deploy the Swedish Library Dashboard to Streamlit Cloud.

## Prerequisites

1. A GitHub account with this repository pushed to it
2. A Streamlit Cloud account (sign up at [https://streamlit.io/cloud](https://streamlit.io/cloud))
3. A Hugging Face account with an API key

## Steps to Deploy

### 1. Push Your Code to GitHub

Make sure your code is pushed to a GitHub repository:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/RezaBaza/swedish-library-dashboard.git
git push -u origin main
```

### 2. Connect to Streamlit Cloud

1. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click on "New app"
3. Connect your GitHub account if you haven't already
4. Select the repository, branch, and main file path (`streamlit_app.py`)

### 3. Configure Secrets

You'll need to add your Hugging Face API key as a secret:

1. In your app settings, find the "Secrets" section
2. Add the following in the text area:

```toml
[general]
HUGGINGFACE_API_KEY = "your_api_key_here"
```

3. Click "Save"

### 4. Advanced Settings (Optional)

You may want to adjust:

- Python version (3.9+ recommended)
- Package dependencies (should be automatically detected from requirements.txt)
- Memory/CPU allocation if your app needs more resources

### 5. Deploy

Click "Deploy" and wait for your app to build and deploy. This may take a few minutes.

### 6. Troubleshooting

If you encounter issues:

1. Check the build logs for errors
2. Ensure all dependencies are correctly listed in requirements.txt
3. Verify that your .env file is not included in the repository (it should be in .gitignore)
4. Make sure your Hugging Face API key is correctly set in the Streamlit Cloud secrets

## Updating Your App

Any new commits pushed to the connected branch will trigger a redeployment of your app.

## Custom Domain (Optional)

If you have a Streamlit Cloud paid plan, you can set up a custom domain for your app in the app settings. 