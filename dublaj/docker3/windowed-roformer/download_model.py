"""
Скачивает модель mbr-win10-sink8.ckpt из smulelabs/windowed-roformer
в volume (/models). Запускается один раз; при повторном запуске
пропускает скачивание, если файл уже на месте.

Источник: https://huggingface.co/smulelabs/windowed-roformer
"""

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "smulelabs/windowed-roformer"
FILENAME = "mbr-win10-sink8.ckpt"
TARGET = Path("/models") / FILENAME


def main() -> None:
    if TARGET.exists() and TARGET.stat().st_size > 0:
        print(f"✓ {FILENAME} — уже скачан, пропускаем", flush=True)
        return

    token = os.environ.get("HF_TOKEN") or None
    print(f"⏳ Скачиваем {REPO_ID}/{FILENAME} ...", flush=True)
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir="/models",
            token=token,
        )
        size_gb = Path(path).stat().st_size / 1024**3
        print(f"✓ Готово: {path} ({size_gb:.2f} GB)", flush=True)
    except Exception as exc:
        print(f"❌ Ошибка скачивания: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
