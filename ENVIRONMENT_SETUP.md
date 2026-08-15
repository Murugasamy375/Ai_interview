# Environment Setup Guide

## Quick Setup (Recommended)

### Step 1: Set Your Groq API Key

**Option A: Using .env file (Easiest)**

1. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

2. Edit `.env` and replace `gsk_YOUR_ACTUAL_API_KEY_HERE` with your actual key:
```
GROK_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

3. Install python-dotenv (if not already installed):
```bash
pip install python-dotenv
```

4. Run the app normally:
```bash
python -m uvicorn app.main:app --reload
```

**✅ That's it! No more API key prompts!**

---

**Option B: Set System Environment Variable (Windows PowerShell)**

```powershell
# Set for current session only:
$env:GROK_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Then run:
python -m uvicorn app.main:app --reload
```

To make it permanent (Windows):
```powershell
# Run as Administrator:
[Environment]::SetEnvironmentVariable("GROK_API_KEY", "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx", "User")

# Then restart PowerShell and run:
python -m uvicorn app.main:app --reload
```

---

**Option C: Set System Environment Variable (Linux/Mac)**

```bash
# Set for current session only:
export GROK_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Then run:
python -m uvicorn app.main:app --reload
```

To make it permanent (Linux/Mac):
```bash
# Add to ~/.bashrc or ~/.zshrc:
export GROK_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Then run:
source ~/.bashrc  # or ~/.zshrc

python -m uvicorn app.main:app --reload
```

---

## Verify Your Setup

Check if your environment variable is set:

**Windows PowerShell:**
```powershell
$env:GROK_API_KEY
# Should output: gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Linux/Mac:**
```bash
echo $GROK_API_KEY
# Should output: gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## What Changed in the Application?

### Before
- API key had to be provided in every request via `X-Grok-API-Key` header
- Frontend constantly asked for the API key
- No way to set it globally

### After
- API key is read from environment variable `GROK_API_KEY`
- API key is used automatically for all LLM operations
- Optional `X-Grok-API-Key` header still works for per-request override
- Much more convenient!

---

## API Key Usage in Requests

### No Header Needed (Recommended)
```bash
# Uses GROK_API_KEY from environment
curl -X POST "http://localhost:8000/start-interview?resume_id=resume.pdf&jd_id=jd.pdf"
```

### With Header Override (Optional)
```bash
# This header overrides the environment variable
curl -X POST "http://localhost:8000/start-interview?resume_id=resume.pdf&jd_id=jd.pdf" \
  -H "X-Grok-API-Key: gsk_different_key"
```

---

## Frontend Usage

The HTML frontend automatically:
1. Uses the API key from environment (backend side)
2. Still allows input in the UI for per-session override
3. Falls back to environment variable if UI is empty

**No more constant API key entry needed!** ✅

---

## Getting Your Groq API Key

If you don't have a Groq API key:

1. Go to https://console.groq.com/keys
2. Sign up or log in
3. Create a new API key
4. Copy the key (starts with `gsk_`)
5. Add to `.env` or system environment

---

## Troubleshooting

### Error: "GROK_API_KEY environment variable is not set!"

**Solution:**
- Make sure you've set the environment variable
- If using .env file, ensure `python-dotenv` is installed: `pip install python-dotenv`
- Check that the `.env` file is in the project root directory
- Restart your terminal/IDE for changes to take effect

### Error: "Invalid API Key"

**Solution:**
- Make sure you copied the full key correctly
- The key should start with `gsk_`
- No extra spaces or characters
- Check https://console.groq.com/keys if the key is still valid

### Still asking for API key in frontend?

**Solution:**
- Backend will use environment variable automatically
- If frontend shows API key input, that's for optional per-request override
- You can leave it blank - backend will use the environment variable
- Or set it in the `.env` file and it will show in the frontend input

---

## Complete Setup Checklist

- [ ] Get Groq API key from https://console.groq.com/keys
- [ ] Create `.env` file from `.env.example`
- [ ] Add API key to `.env`: `GROK_API_KEY=gsk_...`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run app: `python -m uvicorn app.main:app --reload`
- [ ] Open http://localhost:8000
- [ ] Test without entering API key in UI
- [ ] Upload resume and JD
- [ ] Start interview session
- [ ] ✅ Interview should work without manual API key entry!

---

## Questions?

If you encounter issues:
1. Check that environment variable is set: `echo $GROK_API_KEY` (Linux/Mac) or `$env:GROK_API_KEY` (Windows)
2. Verify API key is valid at https://console.groq.com/keys
3. Check application logs for error messages
4. Restart the application after setting environment variables
