# run_local_test.py

import sys, os
import json
from pathlib import Path

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = APP_DIR / "uploads"

sys.path.append(str(ROOT_DIR))

from app.services.processors import process_meeting_video

INPUT_VIDEO = str(UPLOADS_DIR / "sample_meeting.mp4")
OUTPUT_JSON = str(ROOT_DIR / "output_local_test.json")

# -------------------------------------------------------------------
# PRINTING UTILITIES
# -------------------------------------------------------------------

def _safe_get(d: dict, *path, default=None):
    """Safely get nested dict values without KeyError."""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def print_summary(results):
    print("\n" + "=" * 60)
    print("MEETING ANALYSIS SUMMARY")
    print("=" * 60)

    # ---------------- METRICS ----------------
    metrics = _safe_get(results, "insights", "meeting_metrics", default={})
    if metrics:
        print("\n📊 MEETING METRICS:")
        print(f"  Duration: {metrics.get('duration_minutes', 0):.1f} minutes")
        print(f"  Participants: {metrics.get('participant_count', 0)}")
        print(f"  Total Words: {metrics.get('total_words', 0)}")
        print(f"  Mood Changes: {metrics.get('mood_changes', 0)}")
        print(f"  Topics Discussed: {metrics.get('topics_discussed', 0)}")
        print(f"  Action Items: {metrics.get('action_items_count', 0)}")

    # ---------------- PARTICIPANTS ----------------
    participants = results.get("participants", {})
    names = participants.get("detected_names", [])
    print("\n👥 PARTICIPANTS:")
    if names:
        for n in names[:10]:
            print(f"  • {n}")
    else:
        print("  • None detected")

    # ---------------- OVERALL MOOD ----------------
    mood = _safe_get(results, "sentiment", "overall_mood", default={})
    if mood:
        print("\n😊 OVERALL MOOD:")
        print(f"  Dominant Emotion: {mood.get('dominant_emotion', 'N/A')}")
        facial_dist = mood.get("facial_distribution", {})
        if facial_dist:
            print("  Facial Emotions:")
            for emo, pct in sorted(facial_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"    • {emo}: {pct}%")

    # ---------------- KEY MOMENTS ----------------
    key_moments = _safe_get(results, "timeline", "key_moments", default=[])
    if key_moments:
        print("\n⚡ KEY MOOD CHANGES:")
        for i, m in enumerate(key_moments[:5], 1):
            ts = m.get("timestamp", 0)
            print(f"  {i}. At {int(ts//60)}:{int(ts%60):02d}")
            if "from_emotion" in m:
                print(f"     {m['from_emotion']} → {m['to_emotion']}")

    # ---------------- ACTION ITEMS (GEMINI) ----------------
    action_items = _safe_get(results, "insights", "action_items", default=[])
    print("\n✅ ACTION ITEMS (Gemini):")
    if action_items:
        for i, item in enumerate(action_items[:10], 1):
            assignee = item.get("assignee") or "Unassigned"
            print(f"  {i}. {item.get('action', '(missing text)')}")
            print(f"     → {item.get('role','action').upper()}  |  {assignee}")
    else:
        print("  • None detected")

    # ---------------- SUMMARY (GEMINI) ----------------
    summary = _safe_get(results, "insights", "summary", default={})
    main_summary = summary.get("main_summary")
    if main_summary:
        print("\n📝 EXECUTIVE SUMMARY (Gemini):\n")
        print(main_summary)
    else:
        print("\n📝 EXECUTIVE SUMMARY:\n  (No summary produced)")

    print("\n" + "=" * 60)


# -------------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MEETING VIDEO ANALYSIS - LOCAL TEST")
    print("=" * 60)

    print(f"\n📂 ROOT_DIR:     {ROOT_DIR}")
    print(f"📂 APP_DIR:      {APP_DIR}")
    print(f"📂 UPLOADS_DIR:  {UPLOADS_DIR}")
    print(f"📹 INPUT_VIDEO:  {INPUT_VIDEO}")
    print(f"📝 OUTPUT_JSON:  {OUTPUT_JSON}")
    print(f"🏃 CWD:          {os.getcwd()}")

    if not os.path.exists(INPUT_VIDEO):
        print(f"\n❌ ERROR: Video not found: {INPUT_VIDEO}")
        sys.exit(1)

    try:
        size_mb = os.path.getsize(INPUT_VIDEO) / (1024*1024)
        print(f"\n📦 File Size: {size_mb:.2f} MB")
        print("\n⏳ Starting analysis...\n")

        result = process_meeting_video(INPUT_VIDEO, save_output=False)

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Output saved to {OUTPUT_JSON}")

        print_summary(result)

        print("\n✨ Analysis complete!\n")

    except Exception as e:
        print(f"\n❌ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
