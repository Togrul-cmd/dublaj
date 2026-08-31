# 🚀 Dublaj on Colab - Quick Start (5 Steps)

## Before You Start

Get these ready:
- [ ] Gemini API Key: https://aistudio.google.com/apikey
- [ ] HuggingFace Token: https://huggingface.co/settings/tokens
- [ ] Accept licenses:
  - [ ] https://huggingface.co/pyannote/speaker-diarization-community-1
  - [ ] https://huggingface.co/pyannote/segmentation-3.0
- [ ] Video file ready to upload

---

## 5-Step Quick Start

### 1️⃣ Open Colab & Upload Notebook (2 minutes)

1. Go to: https://colab.research.google.com/
2. File → Upload notebook
3. Select: `Dublaj_Colab.ipynb`
4. **Runtime → Change runtime type → T4 GPU → Save**

### 2️⃣ Enter API Keys (1 minute)

In Cell 2, replace:
```python
GEMINI_API_KEY = "YOUR_KEY_HERE"
HF_TOKEN = "YOUR_TOKEN_HERE"
```

### 3️⃣ Run Setup Cells (15 minutes)

Click ▶️ on these cells in order:
- Cell 1: Verify GPU ✅
- Cell 2: Set API keys ✅
- Cell 3: Install FFmpeg (30 sec)
- Cell 4: Clone repo (1 min)
- Cell 5: Install Docker (5 min)
- Cell 6: Start services (10 min) ⏱️
- Cell 7: Wait for ready (auto-checks)

**Wait for "All services ready" before continuing!**

### 4️⃣ Upload & Process Video (5-30 minutes)

- Cell 8: Upload your video ✅
- Cell 9: Set target language ✅
- Cell 10: Create config ✅
- Cell 11: **Process video** ⏱️ (main step, takes longest)

### 5️⃣ Download Result (1 minute)

- Cell 12: Download dubbed video ✅

**Done! 🎉**

---

## Processing Time

| Video Length | Processing Time |
|--------------|-----------------|
| 30 seconds | ~2-3 minutes |
| 1 minute | ~3-5 minutes |
| 5 minutes | ~10-20 minutes |
| 10 minutes | ~20-40 minutes |

---

## Troubleshooting Quick Fixes

| Problem | Fix |
|---------|-----|
| No GPU | Runtime → Change runtime type → T4 GPU |
| Service not ready | Check HF token, verify licenses accepted |
| Translation failed | Check Gemini API key |
| Out of memory | Use shorter video (<5 min) |
| Session disconnected | Re-run from Cell 6 (faster) |

---

## Files You Created

| File | Purpose |
|------|---------|
| `Dublaj_Colab.ipynb` | **Upload this to Colab** |
| `COLAB_GUIDE.md` | Detailed instructions |
| `QUICK_START_COLAB.md` | This file (quick reference) |

---

## What Happens Step by Step

```
Upload notebook → Enable GPU → Enter keys
    ↓
Install dependencies (FFmpeg, Docker)
    ↓
Start 5 AI services (Whisper, OmniVoice, etc.)
    ↓
Upload your video
    ↓
Process: Extract → Separate → Diarize → Transcribe 
         → Translate → Synthesize → Mix → Merge
    ↓
Download dubbed video!
```

---

## Target Language Codes

```python
TARGET_LANGUAGE = "ru"  # Russian
TARGET_LANGUAGE = "tr"  # Turkish  
TARGET_LANGUAGE = "es"  # Spanish
TARGET_LANGUAGE = "fr"  # French
TARGET_LANGUAGE = "de"  # German
TARGET_LANGUAGE = "zh"  # Chinese
TARGET_LANGUAGE = "ja"  # Japanese
TARGET_LANGUAGE = "ar"  # Arabic
```

More: `en`, `it`, `pt`, `ko`, `hi`, `az`, `uk` (600+ total)

---

## Cost

- **Colab Free**: $0 (50 GPU hours/week)
- **Colab Pro**: $10/month (more hours, priority)

---

## Next Steps

1. ✅ Get API keys
2. ✅ Upload `Dublaj_Colab.ipynb` to Colab
3. ✅ Enable GPU
4. ✅ Follow the 5 steps above
5. ✅ Enjoy your dubbed video!

**Need help?** Read `COLAB_GUIDE.md` for detailed instructions.

---

**Total time from start to finish: ~20-30 minutes**
(including setup + processing a 5-minute video)

**Happy Dubbing! 🎬🎙️**
