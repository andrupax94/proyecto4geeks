# instalar pip install yt-dlp librosa soundfile tqdm
from src.utils.config import (
    YOUTUBE_CARCRASH_DIR,
    YOUTUBE_URLS_FILE_CARCRASH
)

import subprocess
import sys
import time

YOUTUBE_CARCRASH_DIR.mkdir(parents=True, exist_ok=True)

with open(YOUTUBE_URLS_FILE_CARCRASH) as f:
    urls = [u.strip() for u in f.readlines() if u.strip()]

print(f"Total URLs: {len(urls)}")

for url in urls:

    print(f"\nDescargando -> {url}")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "-x",
        "--audio-format", "wav",

        # SOLO primeros 30 minutos
        "--download-sections", "*00:00:00-00:45:00",
        "--concurrent-fragments", "1",
        "--retries", "10",
        "--fragment-retries", "10",
        "--socket-timeout", "30",

        "--force-ipv4",

        "-o",
        str(YOUTUBE_CARCRASH_DIR / "%(id)s.%(ext)s"),

        url
    ]

    subprocess.run(cmd)
    time.sleep(2)