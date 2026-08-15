# 🚀 Quick Start Guide

## Setup Your Groq API Key (One-Time Only!)

### ⭐ Easiest Method: Using .env File

```bash
# Step 1: Copy the example file
cp .env.example .env

# Step 2: Edit .env and replace YOUR_KEY with your actual Groq API key
# GROK_API_KEY=gsk_YOUR_ACTUAL_API_KEY_HERE

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
python -m uvicorn app.main:app --reload
```

That's it! 🎉 The app will automatically use your Groq API key.

---

## Quick Alternative: Windows PowerShell

```powershell
# Set API key for this session
$env:GROK_API_KEY="gsk_your_actual_key_here"

# Install and run
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

---

## Quick Alternative: Linux/Mac

```bash
# Set API key for this session
export GROK_API_KEY="gsk_your_actual_key_here"

# Install and run
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

---

## Get Your Groq API Key

1. Visit: https://console.groq.com/keys
2. Sign up (free account)
3. Create new API key
4. Copy the key (starts with `gsk_`)
5. Paste into `.env` file or environment variable

---

## Access the Application

Once running:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

---

## What Happens Now?

✅ **Before Setup**: API key had to be entered in the UI every time

✅ **After Setup**: API key is loaded automatically from environment

✅ **Benefit**: Smooth, seamless experience without constant prompts

---

## Verify It Works

1. Open http://localhost:8000 in your browser
2. Upload a resume (PDF or TXT)
3. Upload a job description (PDF or TXT)
4. Click "Compute ATS Score" - no API key prompt!
5. Click "Start Interview" - no API key prompt!

---

## Next Steps

See `ENVIRONMENT_SETUP.md` for detailed setup instructions for different platforms.

See `REFACTORING_SUMMARY.md` for code structure improvements.

---

## Troubleshooting

**Q: Still getting API key prompt?**
A: Make sure GROK_API_KEY is set and restart the application.

**Q: Error: "GROK_API_KEY environment variable is not set"?**
A: Set the environment variable. Use `.env` file method for easiest setup.

**Q: How do I know if it worked?**
A: Check the server logs - should not show API key errors, and frontend should work smoothly.

---

Enjoy! 🎯
