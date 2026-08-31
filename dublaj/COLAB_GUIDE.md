# Running Dublaj on Google Colab - Complete Guide

## Why Google Colab?

✅ **FREE Tesla T4 GPU** (15 GB VRAM)
✅ **No local installation** (no Docker, no space issues)
✅ **Fast processing** (10-20x faster than your CPU)
✅ **Pre-installed Python** and libraries
✅ **100+ GB free storage** per session

---

## Step-by-Step Instructions

### Step 1: Upload the Notebook to Colab

1. **Go to Google Colab**: https://colab.research.google.com/

2. **Upload the notebook**:
   - Click **File** → **Upload notebook**
   - Click **Choose File**
   - Select: `Dublaj_Colab.ipynb` (from your Desktop\dublaj\dublaj folder)
   - Click **Open**

   **OR** drag and drop the file into Colab

3. **Enable GPU** (CRITICAL!):
   - Click **Runtime** → **Change runtime type**
   - Under **Hardware accelerator**, select: **T4 GPU**
   - Click **Save**

---

### Step 2: Get Your API Keys (Before Running Cells)

#### A. Gemini API Key (REQUIRED - for translation)

1. Go to: https://aistudio.google.com/apikey
2. Click **"Create API key"**
3. Copy the key (looks like: `AIzaSy...`)
4. Save it temporarily

#### B. HuggingFace Token (REQUIRED - for PyAnnote)

1. Go to: https://huggingface.co/join (create account if needed)
2. Go to: https://huggingface.co/settings/tokens
3. Click **"New token"**
4. Select type: **"Fine-grained"**
5. Enable permissions:
   - ✅ Read contents of your repos
   - ✅ Read contents of public gated repos you can access
6. Click **"Generate"**
7. Copy token (looks like: `hf_...`)

**IMPORTANT:** Accept model licenses:
1. Visit: https://huggingface.co/pyannote/speaker-diarization-community-1
   - Click **"Accept"** at the top
2. Visit: https://huggingface.co/pyannote/segmentation-3.0
   - Click **"Accept"** at the top

---

### Step 3: Run the Notebook

**IMPORTANT:** Run cells in order (top to bottom)!

#### Cell 1: Verify GPU
```python
!nvidia-smi
```
**Expected:** Should show "Tesla T4" with 15 GB memory

#### Cell 2: Enter API Keys
Replace the placeholders with your actual keys:
```python
GEMINI_API_KEY = "AIzaSy..."  # Your actual Gemini key
HF_TOKEN = "hf_..."  # Your actual HuggingFace token
```
Then run the cell.

#### Cell 3: Install FFmpeg
Click ▶️ (Run cell)
Wait ~30 seconds

#### Cell 4: Clone Repository
Click ▶️
Wait ~1 minute (downloads Dublaj code)

#### Cell 5: Install Docker
Click ▶️
Wait ~5 minutes (installs Docker in Colab)

#### Cell 6: Start AI Services
Click ▶️
Wait ~10-15 minutes (downloads all AI models)

**First time only:** Models download (~15 GB)
**Next time:** Instant (models cached)

#### Cell 7: Wait for Services
Click ▶️
This cell auto-monitors services
Wait until all show "✅ Ready"

#### Cell 8: Upload Video
Click ▶️
Click **"Choose Files"**
Select your video file
Wait for upload to complete

#### Cell 9: Configure Settings
Change target language if needed:
```python
TARGET_LANGUAGE = "ru"  # Russian
# OR
TARGET_LANGUAGE = "tr"  # Turkish
# OR
TARGET_LANGUAGE = "es"  # Spanish
```
Then run cell

#### Cell 10: Create Config
Click ▶️ (auto-creates .env file)

#### Cell 11: Process Video ⏱️ 
**THIS IS THE MAIN PROCESSING STEP**

Click ▶️

**Processing time estimates:**
- 1-minute video: ~3-5 minutes
- 5-minute video: ~10-20 minutes
- 10-minute video: ~20-40 minutes

You'll see progress logs:
```
Job xxx — extracting audio from video
Job xxx — PyAnnote: speaker diarization
Job xxx — Whisper: transcribing
Job xxx — translating...
Job xxx — OmniVoice synthesis
Job xxx — Seed-VC voice conversion
Job xxx — completed!
```

#### Cell 12: Download Result
Click ▶️

Your dubbed video downloads automatically!

---

## What You'll Get

After processing, you'll have:

1. **Final dubbed video** (`6_final_video.mp4`)
   - Multi-speaker voice cloning
   - Background music preserved
   - Synchronized to original timing

2. **All intermediate files** (in output folder):
   - `1_original_audio.wav` - Extracted audio
   - `2_clean_vocals.wav` - Vocals only (no music)
   - `3_whisper_transcript.txt` - Original transcription
   - `4_translation.txt` - Translation
   - `5_synthesized_audio.wav` - TTS output
   - `README.md` - Processing summary

---

## Processing Multiple Videos

To process another video in the same session:

1. **Upload new video** (run Cell 8 again)
2. **Change language if needed** (Cell 9)
3. **Process** (run Cell 11 again)
4. **Download** (run Cell 12 again)

**No need to restart services!** Models stay loaded.

---

## Troubleshooting

### Problem: "GPU not enabled"
**Solution:**
- Runtime → Change runtime type → T4 GPU → Save
- Runtime → Restart runtime

### Problem: "Service not ready" after 10 minutes
**Solution:**
- Run troubleshooting cell to check logs
- Most common: HuggingFace token not accepted
- Check licenses were accepted

### Problem: "Out of memory"
**Solution:**
- Use shorter video (<5 minutes)
- Runtime → Restart runtime (clears memory)
- Try again with smaller file

### Problem: Session disconnected
**Solution:**
- Don't worry! Models stay cached
- Just re-run from Step 6 (much faster)
- Can take a few minutes to reconnect

### Problem: "Translation failed"
**Solution:**
- Check Gemini API key is correct
- Check key has not expired
- Try refreshing the key

### Problem: "PyAnnote failed"
**Solution:**
- Verify HuggingFace token is correct
- Verify licenses were accepted:
  - https://huggingface.co/pyannote/speaker-diarization-community-1
  - https://huggingface.co/pyannote/segmentation-3.0
- Create new token with correct permissions

---

## Colab Limitations

### Free Tier:
- ✅ 12-hour max session
- ✅ ~50 GPU hours per week
- ✅ May disconnect after 90 min idle
- ✅ Files deleted when session ends

### Tips:
1. **Download results immediately** (don't leave them)
2. **Process one video at a time**
3. **Keep browser tab active** (prevents disconnect)
4. **Upgrade to Colab Pro** if you need:
   - Longer sessions (24 hours)
   - More GPU time
   - Priority access
   - Cost: $10/month

---

## Performance Comparison

| Hardware | 5-Minute Video | 10-Minute Video |
|----------|----------------|-----------------|
| **Your local CPU** | 2-4 hours ⚠️ | 4-8 hours ⚠️ |
| **Your GPU (710M)** | 1-2 hours ⚠️ | 2-4 hours ⚠️ |
| **Colab T4 GPU** | 10-15 min ✅ | 20-30 min ✅ |
| **Cloud A100 GPU** | 3-5 min 🚀 | 6-10 min 🚀 |

---

## Supported Languages

Common codes:
- `ru` - Russian
- `en` - English
- `tr` - Turkish
- `az` - Azerbaijani
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic
- `hi` - Hindi
- `uk` - Ukrainian

**600+ languages total!** Full list at: http://localhost:8200/languages (when services running)

---

## Cost Comparison

| Option | Setup Cost | Per Video | Speed |
|--------|------------|-----------|-------|
| **Colab Free** | $0 | $0 | 10-15 min ✅ |
| **Colab Pro** | $10/month | $0 | 10-15 min ✅ |
| **Vast.ai** | $5 credit | $0.50-1 | 10-15 min |
| **Your PC** | $0 | $0 | 2-4 hours ⚠️ |

**Best choice: Start with Colab Free!**

---

## Advanced: Using Colab Secrets (More Secure)

Instead of entering API keys in code:

1. Click **🔑 icon** in left sidebar
2. Click **"Add new secret"**
3. Add secrets:
   - Name: `GEMINI_API_KEY`, Value: your key
   - Name: `HF_TOKEN`, Value: your token
4. Enable access for notebook

Then in Cell 2, use this code instead:
```python
from google.colab import userdata
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
HF_TOKEN = userdata.get('HF_TOKEN')
```

---

## Next Steps

1. ✅ Upload notebook to Colab
2. ✅ Get API keys
3. ✅ Enable GPU
4. ✅ Run all cells in order
5. ✅ Wait for processing
6. ✅ Download result
7. ✅ Process more videos!

---

## Tips for Best Results

### Video Quality:
- ✅ Use videos with clear audio
- ✅ Minimize background noise
- ✅ Multiple speakers detected automatically
- ✅ Keep original video quality high

### Processing Speed:
- ✅ Shorter videos process faster
- ✅ Split long videos into parts
- ✅ Process multiple short videos in one session

### Quality vs Speed:
Current settings are optimized for quality. To speed up:
- Reduce `SEEDVC_MAX_PARALLEL` to 1 (saves VRAM)
- Disable vocal separation (lower quality)
- Use simpler translation provider

---

## Getting Help

**Check logs if errors occur:**
```python
!docker logs dublaj-whisper
!docker logs dublaj-omnivoice
!docker logs dublaj-pyannote-diarization
!docker logs dublaj-seedvc
```

**Common issues:**
1. GPU not enabled → Check runtime type
2. Services not starting → Check logs
3. Translation failing → Check Gemini key
4. PyAnnote failing → Check HF token + licenses

**Still stuck?**
- Check GitHub issues: https://github.com/Izahat/dublaj/issues
- Review README.md for details

---

## FAQ

**Q: How long can I use Colab for free?**
A: ~50 GPU hours/week, 12-hour sessions

**Q: Do files persist between sessions?**
A: No, download everything before closing

**Q: Can I use my own GPU locally?**
A: Yes, but you need 40+ GB space + modern GPU

**Q: What's the video size limit?**
A: Colab: ~100 GB storage, but upload time matters

**Q: Can I process 1-hour videos?**
A: Yes, but will take ~2-3 hours to process

**Q: Is my video data private?**
A: Colab: runs in Google servers
   Gemini API: processes text (translation)
   Consider privacy when uploading sensitive content

**Q: Can I use different translation provider?**
A: Yes, change `TRANSLATION_PROVIDER` in Cell 10:
   - `gemini` (free, cloud)
   - `llamacpp` (local, slower, no API key needed)

---

**You're all set! Upload the notebook and start dubbing! 🎬**
