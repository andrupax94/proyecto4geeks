# pip install yt-dlp tqdm
from src.utils.config import (
    YOUTUBE_FIRE_VIDEO_DIR,
    YOUTUBE_URLS_FILE_FIRE,
    TARGET_SR,
)

import subprocess
import sys
import time
from tqdm import tqdm

YOUTUBE_FIRE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

with open(YOUTUBE_URLS_FILE_FIRE, encoding="utf-8") as f:
    urls = [u.strip() for u in f.readlines() if u.strip()]

print(f"🔥 Total URLs: {len(urls)}")

for url in tqdm(urls):

    print(f"\n⬇️ Descargando audio -> {url}")

    cmd = [
        sys.executable, "-m", "yt_dlp",

        "-f", "bestaudio/best",

        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",

        "--download-sections", "*00:00:00-00:30:00",

        # 🔥 anti-bloqueo youtube
        "--concurrent-fragments", "1",
        "--retries", "10",
        "--fragment-retries", "10",
        "--socket-timeout", "30",
        "--force-ipv4",

        # normalizar audio
        "--postprocessor-args",
        f"ffmpeg:-ar {TARGET_SR} -ac 1",

        "-o",
        str(YOUTUBE_FIRE_VIDEO_DIR / "%(id)s.%(ext)s"),

        url
    ]

    subprocess.run(cmd)

    # 🔥 evita bloqueo por scraping
    time.sleep(2)

print("\n✅ Descarga finalizada")