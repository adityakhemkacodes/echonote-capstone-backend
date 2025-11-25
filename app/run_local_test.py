import sys, os
import json
from pathlib import Path

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------

# ROOT_DIR = project root: .../echonote_final_backend
ROOT_DIR = Path(__file__).resolve().parent.parent
# APP_DIR = .../echonote_final_backend/app
APP_DIR = Path(__file__).resolve().parent
# UPLOADS_DIR = .../echonote_final_backend/app/uploads
UPLOADS_DIR = APP_DIR / "uploads"

# Add project root to Python path
sys.path.append(str(ROOT_DIR))

from app.services.processors import process_meeting_video

# Video path: .../echonote_final_backend/app/uploads/sample_meeting.mp4
INPUT_VIDEO = str(UPLOADS_DIR / "sample_meeting.mp4")
# Output JSON in project root
OUTPUT_JSON = str(ROOT_DIR / "output_local_test.json")


def print_summary(results):
    """Print a formatted summary of the results"""
    print("\n" + "="*60)
    print("MEETING ANALYSIS SUMMARY")
    print("="*60)
    
    # Metrics
    if 'insights' in results and 'meeting_metrics' in results['insights']:
        metrics = results['insights']['meeting_metrics']
        print("\n📊 MEETING METRICS:")
        print(f"  Duration: {metrics.get('duration_minutes', 0):.1f} minutes")
        print(f"  Participants: {metrics.get('participant_count', 0)}")
        print(f"  Total Words: {metrics.get('total_words', 0)}")
        print(f"  Mood Changes: {metrics.get('mood_changes', 0)}")
        print(f"  Topics Discussed: {metrics.get('topics_discussed', 0)}")
        print(f"  Action Items: {metrics.get('action_items_count', 0)}")
    
    # Participants
    if 'participants' in results:
        print("\n👥 PARTICIPANTS:")
        names = results['participants'].get('detected_names', [])
        if names:
            for name in names[:10]:
                print(f"  • {name}")
        else:
            print("  • No names detected")
    
    # Overall Mood
    if 'sentiment' in results and 'overall_mood' in results['sentiment']:
        mood = results['sentiment']['overall_mood']
        print("\n😊 OVERALL MOOD:")
        print(f"  Dominant Emotion: {mood.get('dominant_emotion', 'N/A')}")
        
        facial_dist = mood.get('facial_distribution', {})
        if facial_dist:
            print("  Facial Emotions:")
            for emotion, pct in sorted(facial_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"    • {emotion}: {pct}%")
    
    # Key Moments
    if 'timeline' in results and 'key_moments' in results['timeline']:
        key_moments = results['timeline']['key_moments']
        if key_moments:
            print("\n⚡ KEY MOOD CHANGES:")
            for i, moment in enumerate(key_moments[:5], 1):
                timestamp = moment.get('timestamp', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                print(f"  {i}. At {minutes}:{seconds:02d}")
                if 'from_emotion' in moment:
                    print(f"     {moment['from_emotion']} → {moment['to_emotion']}")
    
    # Action Items
    if 'insights' in results and 'action_items' in results['insights']:
        actions = results['insights']['action_items']
        if actions:
            print("\n✅ ACTION ITEMS:")
            for i, item in enumerate(actions[:10], 1):
                print(f"  {i}. {item['action']}")
    
    # Summary
    if 'insights' in results and 'summary' in results['insights']:
        summary = results['insights']['summary']
        if isinstance(summary, dict) and 'main_summary' in summary:
            print("\n📝 MEETING SUMMARY:")
            print(f"  {summary['main_summary']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MEETING VIDEO ANALYSIS - LOCAL TEST")
    print("="*60)

    # Debug: show where we are looking
    print(f"\n📂 ROOT_DIR:     {ROOT_DIR}")
    print(f"📂 APP_DIR:      {APP_DIR}")
    print(f"📂 UPLOADS_DIR:  {UPLOADS_DIR}")
    print(f"📹 INPUT_VIDEO:  {INPUT_VIDEO}")
    print(f"📝 OUTPUT_JSON:  {OUTPUT_JSON}")
    print(f"🏃 CWD:          {os.getcwd()}")
    
    # Check if input video exists
    if not os.path.exists(INPUT_VIDEO):
        print(f"\n❌ ERROR: Video file not found: {INPUT_VIDEO}")
        print(f"\nPlease ensure you have a video file at that path.")
        sys.exit(1)
    
    try:
        file_size = os.path.getsize(INPUT_VIDEO) / (1024*1024)
        print(f"\n📹 Input Video: {INPUT_VIDEO}")
        print(f"📦 File Size: {file_size:.2f} MB")
        print(f"\n⏳ Starting analysis... This may take several minutes.\n")
        
        # Process the video
        result = process_meeting_video(INPUT_VIDEO, save_output=False)
        
        # Save additional output
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Additional output saved to: {OUTPUT_JSON}")
        
        # Print formatted summary
        print_summary(result)
        
        print("\n✨ Analysis complete! Check the output files for detailed results.\n")
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        sys.exit(1)
    
    except ImportError as e:
        print(f"\n❌ ERROR: Missing dependency - {e}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        print("\nAnd download spaCy model:")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR: Processing failed")
        print(f"Error: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nTips:")
        print("  - Ensure all dependencies are installed")
        print("  - Check that your video file is valid")
        print("  - Try with a shorter video first (< 5 minutes)")
        sys.exit(1)
