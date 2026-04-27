"""
Sequential queue downloader — reads veo_queue.json and downloads each game.
Skips files that already exist. Updates status in the queue file as it goes.

Usage: python download_queue.py
"""

import json, os, subprocess, sys

FFMPEG   = r"C:\Users\cmaha\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
QUEUE_FILE = "veo_queue.json"


def download(stream_url: str, output: str, label: str):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    print(f"\n[Download] {label}")
    print(f"  -> {output}")
    print(f"  Source: {stream_url[:80]}")
    print(f"  This may take 10-20 minutes...\n")

    result = subprocess.run(
        [FFMPEG, "-y", "-i", stream_url, "-c", "copy", output],
        text=True
    )
    return result.returncode == 0


def main():
    if not os.path.exists(QUEUE_FILE):
        print(f"ERROR: {QUEUE_FILE} not found. Run capture_veo_urls.py first.")
        sys.exit(1)

    with open(QUEUE_FILE) as f:
        queue = json.load(f)

    pending = [q for q in queue if q.get("status") != "done"]
    print(f"[Queue] {len(pending)} games to download\n")

    for item in queue:
        if item.get("status") == "done":
            print(f"[Skip] {item['label']} — already downloaded")
            continue

        if os.path.exists(item["output"]):
            size_mb = os.path.getsize(item["output"]) / (1024*1024)
            if size_mb > 100:
                print(f"[Skip] {item['label']} — file exists ({size_mb:.0f} MB)")
                item["status"] = "done"
                continue

        ok = download(item["stream_url"], item["output"], item["label"])
        item["status"] = "done" if ok else "failed"

        # Save progress after each download
        with open(QUEUE_FILE, "w") as f:
            json.dump(queue, f, indent=2)

        if ok:
            size_mb = os.path.getsize(item["output"]) / (1024*1024)
            print(f"[Done] {item['label']} saved — {size_mb:.0f} MB")
        else:
            print(f"[Failed] {item['label']} — check stream URL, may have expired")

    print(f"\n[All done] Run: python run_highlight.py --video examples/<game>.mp4 --jersey 11")


if __name__ == "__main__":
    main()
