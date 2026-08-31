#!/usr/bin/env python3
"""
Seed-VC Voice Conversion API Server (Singing F0 Model)

Модель: seed-uvit-whisper-base (200M) — F0 conditioned для пения
- 44100 Hz sample rate
- F0 (pitch) conditioning — перенос высоты голоса
- RMVPE pitch extractor
- BigVGAN vocoder

Zero-shot voice conversion:
- source_audio: аудио для преобразования
- reference_audio: аудио для клонирования голоса
- diffusion_steps: качество (по умолчанию 25, 30-50 для лучшего)
- length_adjust: 1.0 = нормальная скорость, <1.0 быстрее, >1.0 медленнее
- inference_cfg_rate: 0.7 по умолчанию
- auto_f0_adjust: автоподстройка высоты (True для лучшего сходства)
- pitch_shift: сдвиг высоты в полутонах (-24 до +24)

API:
  POST /voice_convert — voice conversion
  GET  /health — проверка статуса
"""

import os
import sys
import tempfile
import uuid
import warnings
from pathlib import Path

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
warnings.simplefilter('ignore')

import torch
import torchaudio
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

# Добавляем путь к seed-vc модулям
sys.path.insert(0, '/workspace')

from modules.commons import build_model, load_checkpoint, recursive_munch, str2bool
import yaml
from hf_utils import load_custom_model_from_hf

app = FastAPI(
    title="Seed-VC Voice Conversion API (F0 Base)",
    description="Zero-shot voice conversion with F0 conditioned model",
    version="1.0.0"
)

# Глобальные переменные для модели
model = None
semantic_fn = None
vocoder_fn = None
campplus_model = None
to_mel = None
mel_fn_args = None
f0_fn = None
sr = None
hop_length = None
max_context_window = None
overlap_wave_len = None
overlap_frame_len = 16
device = None
fp16 = True


def load_models():
    """Загрузка Seed-VC F0 Base модели (200M)."""
    global sr, hop_length, fp16
    global model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args, f0_fn

    print("Loading Seed-VC F0 Base model (200M)...")

    # Загружаем F0 conditioning конфиг и чекпоинт
    dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
        "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
    )
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]  # 44100

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # CAMPPlus speaker embedding
    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    # BigVGAN vocoder
    from modules.bigvgan import bigvgan
    bigvgan_name = model_params.vocoder.name
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)
    vocoder_fn = bigvgan_model

    # Whisper semantic encoder
    from transformers import AutoFeatureExtractor, WhisperModel
    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def semantic_fn(waves_16k):
        ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                               return_tensors="pt",
                                               return_attention_mask=True)
        ori_input_features = whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
        return S_ori

    # Mel spectrogram
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    # RMVPE pitch extractor
    from modules.rmvpe import RMVPE
    rmvpe_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
    rmvpe = RMVPE(rmvpe_path, is_half=fp16, device=device)
    f0_fn = rmvpe.infer_from_audio

    print(f"Seed-VC F0 Base loaded. SR={sr}, Hop={hop_length}, Model=200M")
    return model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args, f0_fn


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@torch.no_grad()
def voice_convert(source_path: str, reference_path: str, diffusion_steps: int = 25,
                  length_adjust: float = 1.0, inference_cfg_rate: float = 0.7,
                  auto_f0_adjust: bool = True, pitch_shift: int = 0) -> str:
    """
    Выполняет voice conversion с F0 conditioning.
    Возвращает путь к выходному WAV файлу.
    """
    global model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args, f0_fn
    global sr, hop_length, max_context_window, overlap_wave_len

    # Load audio
    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(reference_path, sr=sr)[0]

    # Process audio (max 25s для reference)
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    # Resample to 16kHz для Whisper
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    # Handle long audio (>30s) с chunking
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = to_mel(source_audio.to(device).float())
    mel2 = to_mel(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # F0 extraction (pitch) — КЛЮЧЕВОЕ для этой модели
    F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
    F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

    if device.type == "mps":
        F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]
    else:
        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

    voiced_F0_ori = F0_ori[F0_ori > 1]
    voiced_F0_alt = F0_alt[F0_alt > 1]

    log_f0_alt = torch.log(F0_alt + 1e-5)
    voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
    voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
    median_log_f0_ori = torch.median(voiced_log_f0_ori)
    median_log_f0_alt = torch.median(voiced_log_f0_alt)

    # shift alt log f0 level to ori log f0 level
    shifted_log_f0_alt = log_f0_alt.clone()
    if auto_f0_adjust:
        shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
    shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    if pitch_shift != 0:
        shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)

    # Length regulation с F0
    cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, codes, commitment_loss, codebook_loss = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    interpolated_shifted_f0_alt = torch.nn.functional.interpolate(
        shifted_f0_alt.unsqueeze(1), size=cond.size(1), mode='nearest').squeeze(1)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        chunk_f0 = interpolated_shifted_f0_alt[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style2, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate
            )
            vc_target = vc_target[:, :, mel2.size(-1):]

        vc_wave = vocoder_fn(vc_target.float()).squeeze().cpu()
        if vc_wave.ndim == 1:
            vc_wave = vc_wave.unsqueeze(0)

        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()

    # Обрезаем первые 60мс — стартовый артефакт CFM/BigVGAN (щелчок "к"/"п"
    # в начале генерации: модель стартует из шума и первые мс нестабильны)
    trim_samples = int(sr * 0.06)
    if vc_wave.size(-1) > trim_samples:
        vc_wave = vc_wave[:, trim_samples:]

    # Save output
    output_path = f"/tmp/seedvc_output_{uuid.uuid4().hex[:8]}.wav"
    torchaudio.save(output_path, vc_wave.cpu(), sr)
    return output_path


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске."""
    global device, max_context_window, overlap_wave_len, sr, hop_length

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    load_models()

    # streaming and chunk processing related params
    max_context_window = sr // hop_length * 30
    overlap_wave_len = overlap_frame_len * hop_length

    print(f"Seed-VC F0 Base ready. Device: {device}")


@app.get("/health")
async def health():
    """Проверка здоровья сервиса."""
    return {
        "status": "healthy",
        "model": "Seed-VC F0 Base (200M)",
        "device": str(device) if device else "cpu"
    }


@app.post("/voice_convert")
async def voice_convert_api(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    diffusion_steps: int = Form(100),
    length_adjust: float = Form(1.0),
    inference_cfg_rate: float = Form(0.8),
    auto_f0_adjust: bool = Form(True),
    f0_condition: bool = Form(True),
    pitch_shift: int = Form(0),
):
    """
    Voice conversion endpoint с F0 conditioning (ВСЕГДА ВКЛЮЧЕНО).

    Параметры (по умолчанию МАКСИМАЛЬНОЕ СХОДСТВО):
    - source: аудио для преобразования (WAV/MP3)
    - reference: reference audio для клонирования голоса
    - diffusion_steps: 100 (макс качество, было 25)
    - length_adjust: 1.0
    - inference_cfg_rate: 0.8 (было 0.7)
    - auto_f0_adjust: TRUE (автоподстройка высоты)
    - f0_condition: TRUE (F0 модель whisper-base ВСЕГДА)
    - pitch_shift: 0
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / f"source_{uuid.uuid4().hex[:8]}.wav"
        ref_path = Path(tmpdir) / f"ref_{uuid.uuid4().hex[:8]}.wav"

        source_content = await source.read()
        ref_content = await reference.read()

        with open(source_path, "wb") as f:
            f.write(source_content)
        with open(ref_path, "wb") as f:
            f.write(ref_content)

        try:
            # F0 conditioning ВСЕГДА включено (эта модель работает ТОЛЬКО с F0)
            output_path = voice_convert(
                source_path=str(source_path),
                reference_path=str(ref_path),
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=inference_cfg_rate,
                auto_f0_adjust=True,    # ВСЕГДА True
                pitch_shift=pitch_shift,
            )
            return FileResponse(
                path=output_path,
                media_type="audio/wav",
                filename="voice_converted.wav"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "service": "Seed-VC F0 Base (Singing) Voice Conversion",
        "model": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
        "model_size": "200M",
        "sample_rate": 44100,
        "endpoints": {
            "POST /voice_convert": "voice conversion with F0 conditioning",
            "GET /health": "health check"
        }
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)