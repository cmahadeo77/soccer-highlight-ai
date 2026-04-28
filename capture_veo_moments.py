"""
Veo Player Moments Capture
Logs into Veo, navigates to the player-moments view for a match,
intercepts the internal API response that powers the moments timeline,
and saves timestamped moments to JSON.

Usage:
  python capture_veo_moments.py \
    --url "https://app.veo.co/matches/20260425-.../  " \
    --jersey 11 \
    --output moments_apr25.json

If --jersey is omitted, all player moments are captured (unfiltered).
"""

import argparse
import json
import os
import re
import sys
import time
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url",     required=True, help="Veo match URL")
    p.add_argument("--jersey",  default=None,  help="Jersey number to filter (e.g. 11). Omit for all players.")
    p.add_argument("--output",  default="moments.json", help="Output JSON file")
    p.add_argument("--debug",   action="store_true", help="Dump all captured API calls to debug_api_calls.json")
    return p.parse_args()


def _looks_like_moments(data: any) -> bool:
    """Heuristic: does this JSON payload contain moment/timestamp data?"""
    text = json.dumps(data).lower()
    return any(k in text for k in ["timestamp", "start_time", "moments", "player_id", "event_time", "clip"])


def _extract_moments(data: any, target_jersey: str | None) -> list[dict]:
    """
    Walk arbitrary JSON trying to extract moment records.
    Veo's moments API likely returns a list of objects with timestamps.
    """
    moments = []
    text    = json.dumps(data)

    # Try to find arrays that look like moment lists
    def walk(node, depth=0):
        if depth > 10:
            return
        if isinstance(node, list):
            for item in node:
                walk(item, depth + 1)
        elif isinstance(node, dict):
            # Does this dict look like a moment?
            keys = {k.lower() for k in node.keys()}
            has_time = any(k in keys for k in ["timestamp", "start_time", "time", "start", "offset"])
            has_player = any(k in keys for k in ["player", "jersey", "player_id", "athlete"])

            if has_time:
                moment = _normalize_moment(node, target_jersey)
                if moment:
                    moments.append(moment)
            else:
                for v in node.values():
                    walk(v, depth + 1)

    walk(data)
    return moments


def _normalize_moment(node: dict, target_jersey: str | None) -> dict | None:
    """
    Normalize a raw Veo moment dict to our standard format:
    {start_sec, end_sec, moment_type, player_jersey, source}
    """
    # Find start timestamp (seconds)
    start_sec = None
    for key in ["timestamp", "start_time", "time", "start", "offset", "startTime"]:
        val = node.get(key)
        if val is not None:
            try:
                start_sec = float(val)
                # Veo may return milliseconds — normalize to seconds
                if start_sec > 10000:
                    start_sec = start_sec / 1000.0
                break
            except (TypeError, ValueError):
                pass

    if start_sec is None:
        return None

    # Find end timestamp
    end_sec = start_sec + 8.0  # default window if no explicit end
    for key in ["end_time", "end", "duration", "endTime"]:
        val = node.get(key)
        if val is not None:
            try:
                v = float(val)
                if v > 10000:
                    v = v / 1000.0
                # If it's a duration, add to start; if it's an absolute time, use directly
                end_sec = start_sec + v if key == "duration" else v
                break
            except (TypeError, ValueError):
                pass

    # Find jersey/player info
    jersey = None
    for key in ["jersey", "jersey_number", "number", "shirt_number", "shirtNumber"]:
        val = node.get(key)
        if val is not None:
            jersey = str(val).strip()
            break

    # Nested player object
    if jersey is None:
        for pkey in ["player", "athlete"]:
            p = node.get(pkey)
            if isinstance(p, dict):
                for key in ["jersey", "jersey_number", "number", "shirt_number", "shirtNumber"]:
                    val = p.get(key)
                    if val is not None:
                        jersey = str(val).strip()
                        break

    # Filter by jersey if requested
    if target_jersey and jersey and jersey != target_jersey:
        return None

    # Moment type / label
    moment_type = "unknown"
    for key in ["type", "event_type", "label", "category", "name", "tag"]:
        val = node.get(key)
        if val is not None:
            moment_type = str(val).lower().replace(" ", "_")
            break

    return {
        "start_sec":   round(start_sec, 2),
        "end_sec":     round(end_sec, 2),
        "moment_type": moment_type,
        "player_jersey": jersey,
        "source":      "veo",
        "raw":         node,  # keep raw for debugging
    }


def run(url: str, jersey: str | None, output: str, debug: bool) -> list[dict]:
    from playwright.sync_api import sync_playwright

    email    = os.environ.get("VEO_EMAIL")
    password = os.environ.get("VEO_PASSWORD")
    if not email or not password:
        print("ERROR: Set VEO_EMAIL and VEO_PASSWORD in .env")
        sys.exit(1)

    captured_api   = []   # all API responses (for debug)
    moments_raw    = []   # candidate moment payloads

    player_moments_url = url.rstrip("/") + "#/player-moments/"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context()
        page    = context.new_page()

        # ── Intercept all JSON API responses ────────────────────────────────
        def on_response(response):
            url_r = response.url
            ct    = response.headers.get("content-type", "")
            if "json" not in ct and "graphql" not in url_r:
                return
            try:
                body = response.json()
            except Exception:
                return

            entry = {"url": url_r, "status": response.status, "body": body}
            captured_api.append(entry)

            if _looks_like_moments(body):
                print(f"  [API] Candidate moments response: {url_r[:100]}")
                moments_raw.append(body)

        page.on("response", on_response)

        # ── Login ────────────────────────────────────────────────────────────
        print("[Auth] Logging in to Veo...")
        page.goto("https://app.veo.co/login/", wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        page.fill('input[type="email"]', email, timeout=10000)
        page.fill('input[type="password"]', password, timeout=10000)
        page.keyboard.press("Enter")
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(2)

        if "login" in page.url.lower():
            print("[Auth] Login failed — check VEO_EMAIL / VEO_PASSWORD")
            browser.close()
            sys.exit(1)
        print("[Auth] Logged in.\n")

        # ── Navigate to match player-moments view ────────────────────────────
        print(f"[Match] Loading player-moments view...")
        page.goto(player_moments_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(5)   # give SPA time to fetch moments data

        # If jersey filter needed, try to find and activate it in the UI
        if jersey:
            print(f"[Player] Looking for jersey #{jersey} filter...")
            try:
                # Veo player selector: look for input or dropdown with jersey/player search
                selectors = [
                    f"text=#{jersey}",
                    f"[data-jersey='{jersey}']",
                    f"[aria-label*='{jersey}']",
                    "input[placeholder*='player' i]",
                    "input[placeholder*='jersey' i]",
                    "input[placeholder*='search' i]",
                ]
                found = False
                for sel in selectors:
                    try:
                        el = page.query_selector(sel)
                        if el:
                            el.click()
                            time.sleep(1)
                            if "input" in sel:
                                el.fill(str(jersey))
                                time.sleep(1)
                                # Try pressing Enter or selecting from dropdown
                                page.keyboard.press("Enter")
                                time.sleep(2)
                            found = True
                            print(f"  Used selector: {sel}")
                            break
                    except Exception:
                        pass

                if not found:
                    print(f"  Could not find jersey #{jersey} UI filter — capturing all player moments")
            except Exception as e:
                print(f"  Player filter attempt failed: {e}")

        # Wait for additional API calls after any filter interaction
        time.sleep(5)
        page.wait_for_load_state("networkidle", timeout=10000)
        time.sleep(2)

        browser.close()

    # ── Save debug dump ──────────────────────────────────────────────────────
    if debug:
        with open("debug_api_calls.json", "w") as f:
            json.dump(captured_api, f, indent=2, default=str)
        print(f"[Debug] {len(captured_api)} API calls saved to debug_api_calls.json")

    # ── Extract and normalize moments ────────────────────────────────────────
    all_moments = []
    for payload in moments_raw:
        extracted = _extract_moments(payload, target_jersey=jersey)
        all_moments.extend(extracted)

    # Deduplicate by start_sec within a 2-second window
    all_moments.sort(key=lambda m: m["start_sec"])
    deduped = []
    for m in all_moments:
        if not deduped or m["start_sec"] - deduped[-1]["start_sec"] > 2.0:
            deduped.append(m)

    # Strip raw field before saving
    clean = [{k: v for k, v in m.items() if k != "raw"} for m in deduped]

    with open(output, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"\n[Done] {len(clean)} moments captured -> {output}")
    if jersey:
        jersey_moments = [m for m in clean if m.get("player_jersey") == jersey]
        print(f"  Jersey #{jersey}: {len(jersey_moments)} moments")

    return clean


def main():
    args = parse_args()
    moments = run(
        url=args.url,
        jersey=args.jersey,
        output=args.output,
        debug=args.debug,
    )
    if not moments:
        print("\n[Note] No moments extracted from API.")
        print("Run with --debug to inspect all captured API calls in debug_api_calls.json.")
        print("Share that file so the API response structure can be mapped correctly.")


if __name__ == "__main__":
    main()
