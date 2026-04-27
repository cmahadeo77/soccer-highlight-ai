"""
Batch Veo URL Capture — logs in once, visits each match, captures all CDN stream URLs.
Saves a download queue to veo_queue.json.

Usage: python capture_veo_urls.py
Add match URLs to the MATCHES list below.
"""

import json, os, time
from dotenv import load_dotenv
load_dotenv()

# ── Add match URLs + output filenames here ────────────────────────────────────
MATCHES = [
    {
        "url":    "https://app.veo.co/matches/20260425-mustang-2013g-ecnl-rl-vs-davis-2013g-ecnl-rl-v7700269/",
        "output": "examples/mustang_vs_davis_apr25.mp4",
        "label":  "vs Davis Apr 25",
    },
    {
        "url":    "https://app.veo.co/matches/20260412-untitled-recording-2026-04-12_15-56-58-vce99cc4/",
        "output": "examples/mustang_apr12.mp4",
        "label":  "Apr 12",
    },
    {
        "url":    "https://app.veo.co/matches/20260215-mustang-2013g-ecnl-rl-vs-fury-2013g-vc7212d5/",
        "output": "examples/mustang_vs_fury.mp4",
        "label":  "vs Fury Feb 15",
    },
    {
        "url":    "https://app.veo.co/matches/20251012-mustang-2013g-ecnl-rl-vs-placer-2013g-ecnl-rl-0d64ec2f/",
        "output": "examples/mustang_vs_placer_oct12.mp4",
        "label":  "vs Placer Oct 12",
    },
    {
        "url":    "https://app.veo.co/matches/20250914-mustang-2013g-ecnl-rl-vs-bay-area-surf-ecnl-rl-3fde2efb/",
        "output": "examples/mustang_vs_bay_area_surf_sep14.mp4",
        "label":  "vs Bay Area Surf Sep 14",
    },
    {
        "url":    "https://app.veo.co/matches/20250906-match-mustang-2013g-ecnl-rl-d6f8b6b7/",
        "output": "examples/mustang_sep06.mp4",
        "label":  "Mustang Sep 6",
    },
]

QUEUE_FILE = "veo_queue.json"

def main():
    email    = os.environ.get("VEO_EMAIL")
    password = os.environ.get("VEO_PASSWORD")

    from playwright.sync_api import sync_playwright

    captured_queue = []
    stream_patterns = [".m3u8", "video.mp4", "stream", "playlist", "manifest"]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context()
        page    = context.new_page()

        # ── Login once ────────────────────────────────────────────────────────
        print("[Auth] Logging in to Veo...")
        page.goto("https://app.veo.co/login/", wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)

        page.fill('input[type="email"]', email, timeout=15000)
        page.fill('input[type="password"]', password, timeout=15000)
        page.keyboard.press("Enter")
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(2)

        if "login" in page.url.lower():
            print("[Auth] ERROR: Login failed. Check VEO_EMAIL / VEO_PASSWORD in .env")
            browser.close()
            return

        print(f"[Auth] Logged in. Session active.\n")

        # ── Visit each match ──────────────────────────────────────────────────
        for match in MATCHES:
            print(f"[{match['label']}] Navigating...")
            captured = []

            def on_response(response, c=captured):
                u = response.url
                if any(p in u for p in stream_patterns):
                    if any(x in u for x in ["veocdn", "cdn", "cloudfront", "akamai"]):
                        if u not in c:
                            c.append(u)
                            print(f"  Captured: {u[:90]}")

            page.on("response", on_response)

            try:
                page.goto(match["url"], wait_until="domcontentloaded", timeout=30000)
                time.sleep(4)

                # Try clicking play to trigger stream
                try:
                    video_el = page.query_selector("video")
                    if video_el:
                        video_el.click()
                        time.sleep(3)
                except Exception:
                    pass

                # Also check video src via DOM
                try:
                    src = page.evaluate("() => { const v = document.querySelector('video'); return v ? (v.src || v.currentSrc) : null; }")
                    if src and src not in captured:
                        captured.append(src)
                        print(f"  DOM src: {src[:90]}")
                except Exception:
                    pass

            except Exception as e:
                print(f"  Navigation error (non-fatal): {e}")

            page.remove_listener("response", on_response)

            # Pick best URL — prefer video.mp4 direct, fall back to m3u8
            stream_url = None
            mp4s  = [u for u in captured if "video.mp4" in u]
            m3u8s = [u for u in captured if ".m3u8" in u]
            stream_url = mp4s[0] if mp4s else (m3u8s[0] if m3u8s else (captured[0] if captured else None))

            if stream_url:
                captured_queue.append({
                    "label":      match["label"],
                    "output":     match["output"],
                    "stream_url": stream_url,
                    "status":     "pending",
                })
                print(f"  [OK] {match['label']} -> {stream_url[:80]}\n")
            else:
                print(f"  [SKIP] No stream URL found for {match['label']}\n")

        browser.close()

    # Save queue
    with open(QUEUE_FILE, "w") as f:
        json.dump(captured_queue, f, indent=2)

    print(f"[Done] {len(captured_queue)} URLs captured -> {QUEUE_FILE}")
    print("Run: python download_queue.py  to download all games sequentially.")


if __name__ == "__main__":
    main()
