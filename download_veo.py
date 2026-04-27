"""
Veo Video Downloader
Logs into app.veo.co, navigates to a match, intercepts the HLS/MP4 stream URL,
and downloads the full game video using ffmpeg.

Usage:
  python download_veo.py --url "https://app.veo.co/matches/..." --output examples/game.mp4
"""

import argparse
import os
import re
import subprocess
import sys
import time
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url",    required=True,  help="Veo match URL")
    p.add_argument("--output", default="examples/game.mp4", help="Output file path")
    p.add_argument("--email",  default=os.environ.get("VEO_EMAIL"),    help="Veo account email")
    p.add_argument("--password", default=os.environ.get("VEO_PASSWORD"), help="Veo account password")
    return p.parse_args()


def download_with_ffmpeg(stream_url: str, output_path: str, headers: dict = None):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y"]

    if headers:
        for k, v in headers.items():
            cmd += ["-headers", f"{k}: {v}\r\n"]

    cmd += [
        "-i", stream_url,
        "-c", "copy",
        output_path,
    ]

    print(f"[Download] Downloading to {output_path} ...")
    print(f"[Download] This may take 10-20 minutes for a full game — please wait.\n")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg download failed")
    return output_path


def run(url: str, output: str, email: str, password: str):
    from playwright.sync_api import sync_playwright

    video_url   = None
    auth_token  = None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)  # visible so user can handle 2FA if needed
        context = browser.new_context()
        page    = context.new_page()

        # Capture network requests to find the video stream
        stream_patterns = [".m3u8", ".mp4", "stream", "playlist", "manifest"]
        captured = []

        def on_response(response):
            u = response.url
            if any(p in u for p in stream_patterns):
                if "veo" in u or "cdn" in u or "cloudfront" in u or "akamai" in u:
                    captured.append(u)
                    print(f"[Network] Captured: {u[:100]}")

        page.on("response", on_response)

        # ── Step 1: Log in ────────────────────────────────────────────────
        print("[Auth] Navigating to Veo login...")
        page.goto("https://app.veo.co/login/", wait_until="networkidle", timeout=30000)
        time.sleep(1)

        print("[Auth] Entering credentials...")
        page.fill('input[type="email"], input[name="email"]', email)
        page.fill('input[type="password"], input[name="password"]', password)
        page.keyboard.press("Enter")

        print("[Auth] Waiting for login to complete...")
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(2)

        # Check if login succeeded
        if "login" in page.url.lower():
            print("[Auth] ERROR: Login may have failed. Check VEO_EMAIL and VEO_PASSWORD in .env")
            browser.close()
            return None

        print(f"[Auth] Logged in. Current URL: {page.url}")

        # ── Step 2: Navigate to match ─────────────────────────────────────
        print(f"\n[Match] Navigating to match...")
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(3)

        print(f"[Match] Page loaded: {page.title()}")

        # ── Step 3: Trigger video playback to expose stream URL ───────────
        print("[Video] Looking for video player...")

        # Try clicking play button
        try:
            play_btn = page.query_selector('button[aria-label*="play" i], .play-button, [class*="play"]')
            if play_btn:
                play_btn.click()
                print("[Video] Clicked play button")
            else:
                # Click the video element directly
                video_el = page.query_selector("video")
                if video_el:
                    video_el.click()
                    print("[Video] Clicked video element")
        except Exception as e:
            print(f"[Video] Play click attempt: {e}")

        # Wait for stream URLs to be captured
        print("[Video] Waiting for stream URL to appear in network traffic...")
        for _ in range(15):
            time.sleep(2)
            m3u8_urls = [u for u in captured if ".m3u8" in u]
            mp4_urls  = [u for u in captured if ".mp4" in u and "m3u8" not in u]
            if m3u8_urls or mp4_urls:
                break

        # Also try extracting src from video element directly
        try:
            video_src = page.evaluate("""
                () => {
                    const v = document.querySelector('video');
                    if (!v) return null;
                    return v.src || v.currentSrc || (v.querySelector('source') ? v.querySelector('source').src : null);
                }
            """)
            if video_src and video_src not in captured:
                captured.append(video_src)
                print(f"[Video] Found via DOM: {video_src[:100]}")
        except Exception:
            pass

        browser.close()

    # ── Step 4: Pick the best stream URL ─────────────────────────────────
    m3u8_urls = [u for u in captured if ".m3u8" in u]
    mp4_urls  = [u for u in captured if ".mp4" in u]

    if not captured:
        print("\n[Error] No video stream URL captured.")
        print("The video may require a club subscription or the page structure may have changed.")
        print("Try downloading manually from app.veo.co and placing the MP4 in examples/")
        return None

    # Prefer highest quality m3u8 (master playlist), fall back to MP4
    stream_url = None
    for u in m3u8_urls:
        if "master" in u or "index" in u or "playlist" in u:
            stream_url = u
            break
    if not stream_url and m3u8_urls:
        stream_url = m3u8_urls[0]
    if not stream_url and mp4_urls:
        stream_url = mp4_urls[0]
    if not stream_url:
        stream_url = captured[0]

    print(f"\n[Stream] Using: {stream_url[:120]}")

    # ── Step 5: Download ──────────────────────────────────────────────────
    download_with_ffmpeg(stream_url, output)
    print(f"\n[Done] Video saved to: {output}")
    print(f"[Next] Run: python run_highlight.py --video {output} --jersey 11")
    return output


def main():
    args = parse_args()

    if not args.email or not args.password:
        print("ERROR: Veo credentials required.")
        print("Add VEO_EMAIL and VEO_PASSWORD to your .env file, or pass --email and --password flags.")
        sys.exit(1)

    run(
        url=args.url,
        output=args.output,
        email=args.email,
        password=args.password,
    )


if __name__ == "__main__":
    main()
