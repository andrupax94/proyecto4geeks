# instalar pip install yt-dlp librosa soundfile tqdm
from src.utils.config import (
    YOUTUBE_RAW_DIR,
    YOUTUBE_URLS_FILE
)

import subprocess

YOUTUBE_RAW_DIR.mkdir(parents=True, exist_ok=True)

with open(YOUTUBE_URLS_FILE) as f:
    urls = [u.strip() for u in f.readlines() if u.strip()]

print(f"Total URLs: {len(urls)}")

for url in urls:

    print(f"\nDescargando -> {url}")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",

        # SOLO primeros 30 minutos
        "--download-sections", "*00:00:00-00:30:00",

        "-o",
        str(YOUTUBE_RAW_DIR / "%(id)s.%(ext)s"),

        url
    ]

    subprocess.run(cmd)