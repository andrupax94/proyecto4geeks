from src.utils.config import (
    YOUTUBE_GUNSHOT_VIDEO_DIR,
    YOUTUBE_GUNSHOT_URLS_FILE
)

import subprocess

YOUTUBE_GUNSHOT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

with open(YOUTUBE_GUNSHOT_URLS_FILE) as f:
    urls = [u.strip() for u in f.readlines() if u.strip()]

print(f"Total URLs: {len(urls)}")

for url in urls:

    print(f"\nDescargando -> {url}")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",

        # primeros 50 minutos
        "--download-sections", "*00:00:00-00:50:00",

        "-o",
        str(YOUTUBE_GUNSHOT_VIDEO_DIR / "%(id)s.%(ext)s"),

        url
    ]

    subprocess.run(cmd)