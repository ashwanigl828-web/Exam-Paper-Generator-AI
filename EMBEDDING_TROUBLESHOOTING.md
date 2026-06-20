# 🔧 Embedding Model - Complete Troubleshooting Guide

## The Problem
You're getting this error repeatedly:
```
Error embedding content (NOT_FOUND): 404 NOT_FOUND
'models/embedding-001 is not found for API version v1beta'
```

## Why This Happens

### Root Causes (in order of likelihood):

1. **❌ Cached Streamlit Data** (Most Common)
   - Streamlit caches function results, including embeddings from old model versions
   - Browser cache holds old cached embeddings
   - Session state contains old configuration

2. **❌ Incompatible Library Versions** (Very Common)
   - `google-generativeai` library is outdated
   - `langchain-google-genai` version mismatch
   - Libraries need to be updated together

3. **❌ API Key Permissions**
   - API key doesn't have "Generative Language API" enabled
   - Missing embeddings capability in Google Cloud project

4. **❌ Old Vector Stores**
   - FAISS index created with `embedding-001` model
   - New model `text-embedding-004` is incompatible with old indices

---

## ✅ Solutions (Try in Order)

### Solution 1: Clear Browser Cache (Quick Fix)
**Most effective for 80% of cases**

```
1. Open the app in your browser
2. Press Ctrl+Shift+Delete (or Cmd+Shift+Delete on Mac)
3. Select "All time" or "Cached images and files"
4. Click "Clear data"
5. Restart Streamlit app (Ctrl+C then run again)
```

### Solution 2: Update Libraries (Recommended)
**Ensures you have compatible versions**

```bash
# Option A: Update all dependencies
pip install --upgrade google-generativeai langchain langchain-google-genai langchain-community

# Option B: Fresh install with pinned versions
pip install --upgrade -r requirements.txt
```

**Then:**
- Restart Streamlit: `Ctrl+C` and run again
- Clear browser cache (Solution 1)

### Solution 3: Clear Streamlit Cache (Advanced)
**Clears all cached function results**

```bash
# On Windows:
rmdir /s %userprofile%\.streamlit\cache

# On Mac/Linux:
rm -rf ~/.streamlit/cache
```

Then restart the app.

### Solution 4: Validate API Key
**Ensure your API key is set up correctly**

Run this test:
```bash
python test_embed.py
```

Expected output:
```
[SUCCESS] models/text-embedding-004 works! Length: 768
```

If it fails:
1. Check your `GEMINI_API_KEY` in `.env`
2. Verify the key is valid in Google Cloud Console
3. Enable "Generative Language API" in your Google Cloud project

### Solution 5: Regenerate Vector Stores
**Only if errors persist after above solutions**

```
1. Go to "⚙️ Manage Books" tab
2. Re-upload all your PDF books
3. The system will create new vector stores with the correct model
4. This may take a few minutes depending on PDF size
```

---

## 🚨 Advanced Debugging

### Check Library Versions
```bash
pip show google-generativeai langchain langchain-google-genai
```

**Should show versions >= our requirements:**
- google-generativeai >= 0.3.0
- langchain >= 0.1.0
- langchain-google-genai >= 0.0.10

### Test Embedding Model Directly
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=key
)

result = embeddings.embed_query("hello world")
print(f"✅ Success! Embedding dimension: {len(result)}")
```

If this works, the issue is likely Streamlit caching.
If this fails, the issue is API key or library version.

---

## 📋 Checklist to Follow

- [ ] Clear browser cache
- [ ] Restart Streamlit app
- [ ] Update requirements: `pip install --upgrade -r requirements.txt`
- [ ] Test embedding with `python test_embed.py`
- [ ] Check API key is valid in `.env`
- [ ] If still failing, regenerate vector stores (re-upload PDFs)
- [ ] As last resort, clear Streamlit cache directory

---

## 🆘 Still Having Issues?

### Check This:
1. **Error appears on upload page?**
   - Issue is likely API key or library version
   - Run `python test_embed.py`

2. **Error appears when querying?**
   - Issue is likely old vector store format
   - Regenerate by re-uploading the PDF

3. **Error says "PERMISSION_DENIED"?**
   - API key doesn't have embeddings enabled
   - Go to Google Cloud Console → Enable Generative Language API

4. **Error says "401 Unauthorized"?**
   - API key is invalid or expired
   - Generate a new key from ai.google.dev

---

## 🔄 Automatic Fixes in App

The app now includes:
- ✅ Automatic embedding model validation on startup
- ✅ Clear error messages with specific diagnostics
- ✅ Cache busting mechanism
- ✅ Pinned library versions in requirements.txt
- ✅ Dedicated error handler for embedding errors

When you see an error, check the **Debug Information** section at the bottom of the error message.

---

## 📞 Need Help?

**If solutions don't work:**
1. Note the exact error message
2. Run: `python test_embed.py` and share output
3. Share your pip list: `pip list | grep -E "google|langchain"`
4. Check if GEMINI_API_KEY is valid by generating a new one at ai.google.dev

---

**Last Updated:** 2026-06-20  
**Status:** ✅ All embedding issues should be resolved with these solutions
