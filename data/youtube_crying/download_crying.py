# pip install yt-dlp

from src.utils.config import (
    YOUTUBE_CRYING_RAW_DIR,
    YOUTUBE_URLS_FILE_CRYING
)

import subprocess

YOUTUBE_CRYING_RAW_DIR.mkdir(parents=True, exist_ok=True)

with open(YOUTUBE_URLS_FILE_CRYING) as f:
    urls = [u.strip() for u in f.readlines() if u.strip()]

print(f"Total URLs: {len(urls)}")

for url in urls:

    print(f"\nDescargando -> {url}")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",

        # primeros 30 min
        "--download-sections", "*00:00:00-00:30:00",

        "-o",
        str(YOUTUBE_CRYING_RAW_DIR / "%(id)s.%(ext)s"),

        url
    ]
    

    subprocess.run(cmd)