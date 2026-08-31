"""
Seed-VC Model Downloader

Скачивает все модели Seed-VC в указанную директорию.
Запускается один раз при первом запуске (или если volume пустой).
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, login, snapshot_download


# Куда качаем — совпадает с HF_HUB_CACHE в основном контейнере
MODEL_DIR = Path("/root/.cache/huggingface")

# Файлы для скачивания из Plachta/Seed-VC
PLACHTA_FILES = {
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth": "DiT checkpoint (200M F0 Base model)",
    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml": "F0 Base config",
}

# Файлы из других репо — нужны для работы Seed-VC
# CAMPPlus: извлекает "отпечаток голоса" (тембр) из референсного аудио
# RMVPE: извлекает мелодию интонации (F0 pitch), чтобы сохранить эмоции при конверсии
OTHER_FILES = [
    ("funasr/campplus", "campplus_cn_common.bin", "CAMPPlus speaker embedding"),
    ("lj1995/VoiceConversionWebUI", "rmvpe.pt", "RMVPE pitch extractor"),
]


def download_plachta_seedvc():
    """Скачать модели из Plachta/Seed-VC."""
    print("=" * 60, flush=True)
    print("Downloading Plachta/Seed-VC (F0 Base, 200M)...", flush=True)
    print("=" * 60, flush=True)

    repo_id = "Plachta/Seed-VC"
    for filename, description in PLACHTA_FILES.items():
        target = MODEL_DIR / "models--Plachta--Seed-VC" / "snapshots"

        # Проверка: уже скачано?
        try:
            files = list(target.rglob(filename)) if target.exists() else []
            if files:
                print(f"✓ {filename} — already exists, skipping", flush=True)
                continue
        except Exception:
            pass

        print(f"⏳ Downloading: {filename} ({description})", flush=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(MODEL_DIR),
                token=os.environ.get("HF_TOKEN") or None,
            )
            print(f"✓ {filename} — done", flush=True)
        except Exception as e:
            print(f"✗ Error: {e}", file=sys.stderr, flush=True)
            raise


def download_campplus():
    """Скачать CAMPPlus speaker embedding."""
    print("=" * 60, flush=True)
    print("Downloading funasr/campplus (speaker embedding)...", flush=True)
    print("=" * 60, flush=True)

    repo_id, filename, description = "funasr/campplus", "campplus_cn_common.bin", "CAMPPlus speaker embedding"

    target_dir = MODEL_DIR / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if target_dir.exists() and list(target_dir.rglob(filename)):
        print(f"✓ {filename} — already exists, skipping", flush=True)
        return

    print(f"⏳ Downloading: {filename} ({description})", flush=True)
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(MODEL_DIR),
            token=os.environ.get("HF_TOKEN") or None,
        )
        print(f"✓ {filename} — done", flush=True)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr, flush=True)
        raise


def download_rmvpe():
    """Скачать RMVPE pitch extractor."""
    print("=" * 60, flush=True)
    print("Downloading lj1995/VoiceConversionWebUI (RMVPE)...", flush=True)
    print("=" * 60, flush=True)

    repo_id = "lj1995/VoiceConversionWebUI"
    filename = "rmvpe.pt"

    target_dir = MODEL_DIR / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if target_dir.exists() and list(target_dir.rglob(filename)):
        print(f"✓ {filename} — already exists, skipping", flush=True)
        return

    print(f"⏳ Downloading: {filename} (RMVPE pitch extractor)", flush=True)
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(MODEL_DIR),
            token=os.environ.get("HF_TOKEN") or None,
        )
        print(f"✓ {filename} — done", flush=True)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr, flush=True)
        raise


def download_bigvgan():
    """Скачать BigVGAN vocoder (44kHz для F0 Base модели)."""
    print("=" * 60, flush=True)
    print("Downloading nvidia/bigvgan_v2_44khz_128band_512x (vocoder)...", flush=True)
    print("=" * 60, flush=True)

    repo_id = "nvidia/bigvgan_v2_44khz_128band_512x"

    target_dir = MODEL_DIR / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if target_dir.exists() and any(target_dir.rglob("*.pt")):
        print(f"✓ BigVGAN — already exists, skipping", flush=True)
        return

    print(f"⏳ Downloading BigVGAN vocoder (44kHz)...", flush=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(MODEL_DIR),
            token=os.environ.get("HF_TOKEN") or None,
        )
        print(f"✓ BigVGAN — done", flush=True)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr, flush=True)
        raise


def main():
    # Аутентификация через токен — ускоряет скачивание
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print("HuggingFace: authenticated via HF_TOKEN\n", flush=True)
    else:
        print("Warning: HF_TOKEN not set — download may be slow due to rate limits\n", flush=True)

    print(f"Target directory: {MODEL_DIR}\n", flush=True)

    try:
        download_plachta_seedvc()
        download_campplus()
        download_rmvpe()
        download_bigvgan()

        print("\n" + "=" * 60, flush=True)
        print("✅ All Seed-VC models downloaded successfully!", flush=True)
        print("=" * 60, flush=True)
    except Exception as e:
        print(f"\n❌ Download failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()