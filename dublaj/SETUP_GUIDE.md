# Dublaj - Complete Setup Guide for Windows

## Prerequisites

### 1. Install Docker Desktop
1. Download from: https://www.docker.com/products/docker-desktop/
2. Run the installer
3. **IMPORTANT**: Enable WSL 2 backend during installation
4. Restart your computer
5. Start Docker Desktop
6. Verify installation:
   ```powershell
   docker --version
   docker compose version
   ```

### 2. Install NVIDIA Container Toolkit (for GPU support)
**Required if you have NVIDIA GPU (recommended for faster processing)**

1. Install NVIDIA drivers: https://www.nvidia.com/download/index.aspx
2. Install NVIDIA Container Toolkit:
   - Follow guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
3. Verify GPU access:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

### 3. Get API Keys

#### HuggingFace Token (REQUIRED for PyAnnote)
1. Create account: https://huggingface.co/join
2. Create token: https://huggingface.co/settings/tokens
   - Click "New token"
   - Type: **"Fine-grained"**
   - Permissions:
     - ✅ Read contents of your repos
     - ✅ Read contents of public gated repos you can access
3. Accept model licenses:
   - https://huggingface.co/pyannote/speaker-diarization-community-1 → Click "Accept"
   - https://huggingface.co/pyannote/segmentation-3.0 → Click "Accept"

#### Gemini API Key (REQUIRED for translation)
1. Go to: https://aistudio.google.com/apikey
2. Click "Create API key"
3. Copy the key

---

## Setup Instructions

### Step 1: Configure Environment

1. **Copy example config:**
   ```powershell
   cd C:\Users\Admin\Desktop\dublaj\dublaj
   copy .env.example .env
   ```

2. **Edit .env file** (use Notepad or VS Code):
   ```powershell
   notepad .env
   ```

3. **Add your API keys:**
   ```env
   # --- Gemini Translation ---
   GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_KEY_HERE
   
   # --- HuggingFace (for PyAnnote) ---
   HF_TOKEN=YOUR_ACTUAL_HF_TOKEN_HERE
   ```

4. **Verify settings** (recommended defaults):
   ```env
   TRANSLATION_PROVIDER=gemini
   TRANSLATION_MODEL=gemini-2.0-flash
   TTS_PROVIDER=omnivoice
   USE_VOICE_SEPARATION=true
   MAX_CONCURRENT_JOBS=2
   ```

### Step 2: Install Python Dependencies

```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Start Docker Services

**All services must be running before starting the main app.**

#### 3.1 Whisper (Speech Recognition) - Port 8100
```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj\docker3\whisper
docker compose -f docker-compose.whisper.yml up -d
```
Wait for model download (~2GB, first time only). Check logs:
```powershell
docker compose -f docker-compose.whisper.yml logs -f
```
Press Ctrl+C when you see "Uvicorn running"

#### 3.2 OmniVoice (TTS) - Port 8200
```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj\docker3\omnivoice
docker compose -f docker-compose.omnivoice.yml up -d --build
```
Check logs:
```powershell
docker compose -f docker-compose.omnivoice.yml logs -f
```

#### 3.3 Windowed RoFormer (Vocal Separation) - Port 8310
```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj\docker3\windowed-roformer
docker compose -f docker-compose.windowed-roformer.yml up -d --build
```
Check logs:
```powershell
docker compose -f docker-compose.windowed-roformer.yml logs -f
```

#### 3.4 PyAnnote (Speaker Diarization) - Port 8500
```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj\docker3\pyannote
docker compose -f docker-compose.pyannote.yml up -d --build
```
Check logs:
```powershell
docker compose -f docker-compose.pyannote.yml logs -f
```
**Note**: First run downloads models (~3GB), takes 5-10 min

#### 3.5 Seed-VC V1 (Voice Conversion) - Port 8700
```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj\docker3\seedvc
docker compose -f docker-compose.seedvc.yml up -d --build
```
Check logs:
```powershell
docker compose -f docker-compose.seedvc.yml logs -f
```

#### 3.6 (Optional) Gemma E4B Local Translation - Port 8600
**Only if you want offline translation (no Gemini API needed)**
```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj\docker3\gemma_e4b
docker compose -f docker-compose.yml up -d --build
```
Then change in `.env`:
```env
TRANSLATION_PROVIDER=llamacpp
```

### Step 4: Verify All Services Are Running

```powershell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected output:
```
NAMES                           STATUS          PORTS
dublaj-whisper                  Up             0.0.0.0:8100->8000/tcp
dublaj-omnivoice                Up             0.0.0.0:8200->8000/tcp
dublaj-windowed-roformer        Up             0.0.0.0:8310->8000/tcp
dublaj-pyannote-diarization     Up             0.0.0.0:8500->8000/tcp
dublaj-seedvc                   Up             0.0.0.0:8700->8000/tcp
```

### Step 5: Test Services Health

```powershell
# Test each service (should return 200 OK)
curl http://localhost:8100/health
curl http://localhost:8200/health
curl http://localhost:8310/health
curl http://localhost:8500/health
curl http://localhost:8700/health
```

### Step 6: Start Main Application

```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj
python main.py
```

You should see:
```
Starting Dublaj — Video Dubbing Pipeline
Whisper service: http://localhost:8100/v1/audio/transcriptions
Translation model: gemini-2.0-flash (gemini)
TTS service: OmniVoice (http://localhost:8200, voice: male)
All services initialized — ready to accept requests
Uvicorn running on http://0.0.0.0:8000
```

### Step 7: Access the API

Open in browser: **http://localhost:8000/docs**

This opens the interactive API documentation (Swagger UI).

---

## Usage

### Method 1: Web Interface (Easiest)

1. Go to: http://localhost:8000/docs
2. Click on **POST /jobs**
3. Click **"Try it out"**
4. Upload your video file
5. Enter target language (e.g., `ru` for Russian, `tr` for Turkish)
6. Click **"Execute"**
7. Copy the `job_id` from response
8. Use **GET /jobs/{job_id}/status** to check progress
9. When status = "completed", use **GET /jobs/{job_id}/result** to download

### Method 2: Command Line (cURL)

```powershell
# Submit job
curl -X POST "http://localhost:8000/jobs" `
  -F "video=@C:\path\to\your\video.mp4" `
  -F "target_language=ru"

# Check status (replace YOUR_JOB_ID)
curl http://localhost:8000/jobs/YOUR_JOB_ID/status

# Download result
curl -o dubbed_video.mp4 http://localhost:8000/jobs/YOUR_JOB_ID/result
```

### Method 3: Python Script

```python
import requests

# Submit job
files = {'video': open('video.mp4', 'rb')}
data = {'target_language': 'ru'}
response = requests.post('http://localhost:8000/jobs', files=files, data=data)
job_id = response.json()['job_id']
print(f"Job ID: {job_id}")

# Check status
status_response = requests.get(f'http://localhost:8000/jobs/{job_id}/status')
print(status_response.json())

# Download when ready
result = requests.get(f'http://localhost:8000/jobs/{job_id}/result')
with open('dubbed_output.mp4', 'wb') as f:
    f.write(result.content)
```

---

## Supported Languages

Use these codes for `target_language`:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `ru` | Russian | `en` | English | `tr` | Turkish |
| `az` | Azerbaijani | `es` | Spanish | `fr` | French |
| `de` | German | `it` | Italian | `pt` | Portuguese |
| `zh` | Chinese | `ja` | Japanese | `ko` | Korean |
| `ar` | Arabic | `hi` | Hindi | `uk` | Ukrainian |

Full list (600+ languages): http://localhost:8200/languages

---

## Stopping Services

### Stop main app:
Press **Ctrl+C** in the terminal running `python main.py`

### Stop Docker services:
```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj\docker3\whisper
docker compose -f docker-compose.whisper.yml down

cd ..\omnivoice
docker compose -f docker-compose.omnivoice.yml down

cd ..\windowed-roformer
docker compose -f docker-compose.windowed-roformer.yml down

cd ..\pyannote
docker compose -f docker-compose.pyannote.yml down

cd ..\seedvc
docker compose -f docker-compose.seedvc.yml down
```

### Stop all containers at once:
```powershell
docker stop $(docker ps -q)
```

---

## Troubleshooting

### Issue: Docker service not starting

**Check logs:**
```powershell
cd docker3\SERVICE_NAME
docker compose -f docker-compose.*.yml logs -f
```

**Common fixes:**
- Restart Docker Desktop
- Check if port is already in use: `netstat -ano | findstr :8100`
- Rebuild container: `docker compose -f docker-compose.*.yml up -d --build --force-recreate`

### Issue: PyAnnote 403 Forbidden

**Cause**: HuggingFace license not accepted or invalid token

**Fix:**
1. Accept licenses at:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
2. Verify token in `.env` is correct
3. Restart PyAnnote: `docker compose -f docker-compose.pyannote.yml restart`

### Issue: GPU not detected

**Check NVIDIA setup:**
```powershell
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Issue: "Service not ready" error

**Verify all services are healthy:**
```powershell
curl http://localhost:8100/health
curl http://localhost:8200/health
curl http://localhost:8310/health
curl http://localhost:8500/health
curl http://localhost:8700/health
```

Each should return `{"status":"ok"}`

### Issue: Out of memory / GPU OOM

**Reduce concurrent processing in `.env`:**
```env
MAX_CONCURRENT_JOBS=1
SEEDVC_MAX_PARALLEL=1
```

---

## Performance Tips

### Processing Speed
- **GPU**: ~3-5 minutes per minute of video
- **CPU only**: ~10-20 minutes per minute of video

### Optimize for Speed
1. Use shorter videos for testing (<1 min)
2. Reduce concurrent jobs if GPU runs out of memory
3. Use Whisper Turbo (faster model):
   ```powershell
   cd docker3\whisper-turbo
   docker compose -f docker-compose.whisper-turbo.yml up -d --build
   ```

### Disk Space Requirements
- Docker images: ~15-20 GB
- Models (downloaded on first run): ~10-15 GB
- Temp files during processing: ~2-5 GB per job
- **Total**: ~30-40 GB free space recommended

---

## File Locations

### Output Files
All outputs saved to: `C:\Users\Admin\Desktop\dublaj\dublaj\output\{job_id}\`

Contents:
- `1_original_audio.wav` - Extracted audio
- `2_clean_vocals.wav` - Separated vocals
- `3_whisper_transcript.txt/.json` - Transcription
- `4_translation.txt` - Translation
- `4_segments.json` - Per-segment translation
- `5_synthesized_audio.wav` - Final TTS
- `6_final_video.mp4` - **FINAL OUTPUT**
- `README.md` - Summary

### Temporary Files
Located in: `C:\Users\Admin\Desktop\dublaj\dublaj\tmp\{job_id}\`

Automatically cleaned up can be deleted after job completes.

---

## Advanced Configuration

### Use Local Translation (No API Key Required)

1. Start Gemma E4B service
2. Change in `.env`:
   ```env
   TRANSLATION_PROVIDER=llamacpp
   LLAMACPP_BASE_URL=http://localhost:8600
   ```

### Change Voice Gender

In `.env`:
```env
OMNIVOICE_VOICE=female   # or "male"
```

### Disable Vocal Separation (Faster, Lower Quality)

In `.env`:
```env
USE_VOICE_SEPARATION=false
```

### Run Main App in Docker

```powershell
cd C:\Users\Admin\Desktop\dublaj\dublaj
docker compose -f docker3\docker-compose.dublaj.yml up -d
```

Then access API at: http://localhost:8000/docs

---

## Getting Help

### Check Service Status
```powershell
docker ps -a
docker compose -f docker3\SERVICE\docker-compose.*.yml logs --tail=100
```

### View Main App Logs
Main app prints to console where you ran `python main.py`

### Common Log Locations
- Docker logs: `docker logs CONTAINER_NAME`
- Main app: Console output
- Temp files: `tmp/{job_id}/`
- Output: `output/{job_id}/README.md`

---

## Next Steps

1. ✅ Follow setup steps above
2. ✅ Test with short video (<30 seconds)
3. ✅ Check `output/{job_id}/README.md` for results
4. ✅ Try different target languages
5. ✅ Experiment with longer videos

**Happy Dubbing!** 🎬🎙️
