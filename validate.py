"""
Validation Report — review pipeline accuracy before committing to the full reel.
Reads reel_brief.json from a completed pipeline run and generates:
  - validation_report.html  — visual grid of every keyframe with bounding box,
                              event type, score, and confidence for quick human review
  - flagged_clips.json      — low-confidence detections that need manual check

Usage: python validate.py --output output/jersey_11_20260427_135347
"""

import argparse
import json
import os
import base64


def img_to_b64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_report(output_dir: str):
    brief_path = os.path.join(output_dir, "reel_brief.json")
    if not os.path.exists(brief_path):
        print(f"ERROR: No reel_brief.json found in {output_dir}")
        return

    with open(brief_path) as f:
        brief = json.load(f)

    events   = brief.get("events", [])
    jersey   = brief.get("player_jersey", "?")
    total    = len(events)
    flagged  = [e for e in events if e.get("confidence") == "low" or e.get("highlight_score", 10) <= 3]

    print(f"\n[Validate] Jersey #{jersey} — {total} events detected")
    print(f"[Validate] {len(flagged)} low-confidence / low-score events flagged for review")

    # ── HTML report ───────────────────────────────────────────────────────────
    cards = ""
    for e in sorted(events, key=lambda x: x.get("highlight_score", 0), reverse=True):
        kf   = e.get("keyframe_path", "")
        b64  = img_to_b64(kf)
        img_tag = f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:4px">' if b64 else '<div style="background:#222;height:120px;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#666">No keyframe</div>'

        score      = e.get("highlight_score", 0)
        conf       = e.get("confidence", "low")
        event_type = e.get("event_type", "unknown").replace("_", " ").title()
        desc       = e.get("description", "")
        ts         = e.get("timestamp_sec", 0)
        mins       = int(ts // 60)
        secs       = int(ts % 60)

        score_color = "#22c55e" if score >= 7 else "#f59e0b" if score >= 5 else "#ef4444"
        conf_color  = "#22c55e" if conf == "high" else "#f59e0b" if conf == "medium" else "#ef4444"
        border      = "2px solid #ef4444" if conf == "low" or score <= 3 else "2px solid #333"

        clip_name = ""
        for clip in brief.get("top_clips", []):
            if e.get("event_id", "") in clip:
                clip_name = os.path.basename(clip)
                break

        cards += f"""
        <div style="background:#1a1a1a;border:{border};border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:8px">
          {img_tag}
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600;font-size:14px">{event_type}</span>
            <span style="background:{score_color};color:#000;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:700">{score}/10</span>
          </div>
          <div style="font-size:12px;color:#aaa">{mins:02d}:{secs:02d} into game</div>
          <div style="font-size:12px;color:#ccc">{desc}</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap">
            <span style="background:{conf_color}22;color:{conf_color};border:1px solid {conf_color};padding:1px 6px;border-radius:10px;font-size:11px">confidence: {conf}</span>
            {'<span style="background:#ef444422;color:#ef4444;border:1px solid #ef4444;padding:1px 6px;border-radius:10px;font-size:11px">REVIEW</span>' if conf == "low" or score <= 3 else ''}
          </div>
          {'<div style="font-size:11px;color:#666">Clip: ' + clip_name + '</div>' if clip_name else ''}
        </div>"""

    summary_items = ""
    for k, v in brief.get("summary", {}).items():
        summary_items += f'<div style="background:#222;padding:8px 14px;border-radius:6px"><span style="color:#aaa;font-size:12px">{k.replace("_"," ").title()}</span><br><span style="font-size:20px;font-weight:700">{v}</span></div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Validation Report — Jersey #{jersey}</title>
<style>
  body {{ background:#0f0f0f; color:#fff; font-family:system-ui,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:22px; margin:0 0 4px }}
  .subtitle {{ color:#888; font-size:14px; margin-bottom:24px }}
  .summary {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:28px }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px }}
  .warning {{ background:#ef444422; border:1px solid #ef4444; border-radius:8px; padding:12px 16px; margin-bottom:20px; font-size:14px }}
</style>
</head>
<body>
  <h1>Validation Report — Jersey #{jersey}</h1>
  <div class="subtitle">{total} events detected &nbsp;|&nbsp; {brief.get("video_source","")}</div>

  {'<div class="warning">⚠ ' + str(len(flagged)) + ' events flagged for manual review (red border). Check that the correct player is boxed in these frames before including in the reel.</div>' if flagged else ''}

  <div class="summary">{summary_items}</div>

  <h2 style="font-size:16px;margin:0 0 14px">All Events — sorted by highlight score</h2>
  <div class="grid">{cards}</div>

  <div style="margin-top:32px;padding:16px;background:#1a1a1a;border-radius:8px">
    <div style="font-size:14px;font-weight:600;margin-bottom:8px">Recruiting Summary</div>
    <div style="font-size:14px;color:#ccc;line-height:1.6">{brief.get("recruiting_notes","Not yet generated.")}</div>
  </div>
</body>
</html>"""

    report_path = os.path.join(output_dir, "validation_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Save flagged list
    flagged_path = os.path.join(output_dir, "flagged_clips.json")
    with open(flagged_path, "w") as f:
        json.dump(flagged, f, indent=2)

    print(f"\n[Done] Validation report: {report_path}")
    print(f"[Done] Flagged clips:      {flagged_path}")
    print(f"\nOpen the HTML file in your browser to review every detected play.")
    print(f"Red-bordered cards = low confidence or low score — verify the correct player is boxed.")

    return report_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="Pipeline output directory")
    args = p.parse_args()
    build_report(args.output)


if __name__ == "__main__":
    main()
