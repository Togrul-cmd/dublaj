# Running Dublaj on Cloud GPU

Since your local machine lacks GPU and has limited space, consider using cloud GPU services:

## Option 1: Google Colab (FREE GPU)

**Pros:**
- ✅ FREE Tesla T4 GPU (limited hours)
- ✅ No local installation
- ✅ 15 GB RAM, 100+ GB storage
- ✅ Fast processing (5-10x faster than your CPU)

**Cons:**
- ❌ Session timeout after 12 hours
- ❌ Limited to ~50 hours/week on free tier
- ❌ Need to upload videos each time

**Setup:**
1. Go to: https://colab.research.google.com/
2. Create new notebook
3. Runtime → Change runtime type → GPU
4. Install Docker in Colab (possible but complex)

**OR** use pre-made notebook services like:
- **Paperspace Gradient** (free tier with GPU)
- **Kaggle Notebooks** (free GPU, 30h/week)

---

## Option 2: Vast.ai (Pay-per-use GPU)

**Cost:** ~$0.10-0.30/hour for GPU instance

**Pros:**
- ✅ Powerful GPUs (RTX 3090, 4090, etc.)
- ✅ Pay only for usage
- ✅ Fast setup
- ✅ SSH access for Docker

**Cons:**
- ❌ Costs money (but cheap)
- ❌ Requires setup knowledge

**Steps:**
1. Sign up: https://vast.ai/
2. Search for "Docker" template with GPU
3. Rent instance (~$0.20/hour)
4. SSH into instance
5. Run Dublaj Docker containers

**Estimated cost:** $0.50-1.00 per video (5-10 min video)

---

## Option 3: RunPod (GPU Containers)

**Cost:** ~$0.20-0.50/hour

**Pros:**
- ✅ Docker-ready instances
- ✅ Easy to use
- ✅ Pre-configured templates

**Link:** https://www.runpod.io/

---

## Option 4: Lambda Labs (Persistent GPU)

**Cost:** $0.50-1.10/hour

**Pros:**
- ✅ High-end GPUs
- ✅ Persistent storage
- ✅ Easy Docker setup

**Link:** https://lambdalabs.com/

---

## Option 5: Local Cloud Alternative - Use API Services Only

Instead of running the full pipeline, use cloud APIs for heavy tasks:

### Modified Architecture:
```
Local (Your PC):
  - Video processing (FFmpeg) ✅
  - API orchestration ✅

Cloud APIs:
  - Whisper: Use OpenAI Whisper API ($0.006/min)
  - Translation: Gemini API (FREE - you have this)
  - TTS: ElevenLabs API (~$0.30/min)
  - Voice Cloning: Resemble.ai API
```

**Pros:**
- ✅ No Docker needed
- ✅ No GPU needed
- ✅ Much less space (<5 GB)
- ✅ Fast processing

**Cons:**
- ❌ Costs money per video
- ❌ Requires API keys
- ❌ Privacy concerns (data sent to cloud)

**Estimated cost per 5-min video:** ~$2-5

---

## Recommendation for Your Situation

Given your constraints:
- ❌ Only 7 GB free space
- ❌ No GPU
- ❌ Old CPU (2012)

**Best option: Vast.ai or RunPod**
- Rent GPU for $0.20/hour
- Process videos fast
- Pay only when needed
- Total cost: $1-2 per video

**OR**

**Simplest option: Wait and upgrade PC**
- Save up for used GPU (~$100-200 for GTX 1660)
- Add more storage (500GB SSD ~$40)
- Will handle local processing well

---

## Local Testing (If You Still Want to Try)

If you MUST test locally:

1. **Free up 40+ GB space** (see FREE_SPACE_GUIDE.md)
2. **Test with SHORT videos only** (<30 seconds)
3. **Expect 20-30 min processing** for 30s video
4. **Monitor RAM usage** (8GB is barely enough)
5. **Close all other programs** while processing

**Warning:** Your system might:
- Freeze during processing
- Run out of memory
- Take hours for longer videos
- Overheat (old laptop CPU)

---

## Cost Comparison

| Option | Setup Cost | Per-Video Cost | Processing Time (5-min video) |
|--------|------------|----------------|-------------------------------|
| **Your PC** | $0 (need 40GB space) | $0 | **2-4 hours** ⚠️ |
| **Vast.ai** | $5 credit | $0.50-1.00 | **5-10 minutes** ✅ |
| **API Services** | $0 | $2-5 | **10-20 minutes** |
| **Google Colab** | $0 | $0 (limited) | **10-15 minutes** |

---

## Final Recommendation

**If you're serious about video dubbing:**
→ Use **Vast.ai** ($0.20/hour GPU rental)

**If this is just to learn/test:**
→ Free up space and try locally (expect slow performance)

**If you need production quality:**
→ Upgrade hardware (used GPU + more storage) or use cloud
