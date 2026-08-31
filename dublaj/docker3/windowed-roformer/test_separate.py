"""
Тест windowed-roformer сервиса: берёт аудио-файл, шлёт в /separate_json,
сохраняет vocals + instrumental, печатает время работы.

Использование (из корня проекта, venv активирован):
    python docker3/windowed-roformer/test_separate.py tmp/<job_id>/original_audio.wav
"""

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

URL = "http://localhost:8310"


def main() -> None:
    if len(sys.argv) < 2:
        # Если путь не передали — берём первый попавшийся original_audio.wav из tmp/
        candidates = sorted(Path("tmp").glob("*/original_audio.wav"))
        if not candidates:
            print("Передайте путь к аудио: python test_separate.py <file.wav>")
            sys.exit(1)
        audio_path = candidates[-1]
    else:
        audio_path = Path(sys.argv[1])

    print(f"1. Health:    GET {URL}/health")
    health = json.loads(urllib.request.urlopen(f"{URL}/health", timeout=10).read())
    print(f"   → {health}")
    assert health.get("status") == "ok", "Сервис не готов!"

    audio_bytes = audio_path.read_bytes()
    print(f"\n2. Separating: {audio_path} ({len(audio_bytes) / 1e6:.2f} MB)")

    payload = json.dumps({
        "audio_base64": base64.b64encode(audio_bytes).decode(),
        "filename": audio_path.name,
    }).encode()

    req = urllib.request.Request(
        f"{URL}/separate_json",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    elapsed = time.perf_counter() - t0

    vocals = base64.b64decode(resp["vocals_base64"])
    instrumental = base64.b64decode(resp["instrumental_base64"])

    out_voc = Path("tmp") / "wr_test_vocals.wav"
    out_inst = Path("tmp") / "wr_test_instrumental.wav"
    out_voc.write_bytes(vocals)
    out_inst.write_bytes(instrumental)

    print(f"\n3. Готово за {elapsed:.1f} сек:")
    print(f"   🗣️ vocals:       {len(vocals) / 1e6:.2f} MB → {out_voc}")
    print(f"   🎵 instrumental: {len(instrumental) / 1e6:.2f} MB → {out_inst}")
    print("\nПослушай файлы и сравни разделение!")


if __name__ == "__main__":
    main()
