# app/services/routes.py

from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import os
import traceback

from .file_handler import FileHandler
from .processors import MeetingProcessor

api = Blueprint("api", __name__)
file_handler = FileHandler(upload_folder="app/uploads")
processing_status = {}


@api.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {"status": "healthy", "service": "meeting-analysis-api", "version": "1.0.0"}
    ), 200


@api.route("/upload", methods=["POST"])
def upload_video():
    """
    Upload:
      - a meeting video (old flow), OR
      - a zip of the full Zoom folder (new flow)

    Returns: file_id and basic file information
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["file"]

        # Save file (may extract zip and return the main meeting video path)
        success, filepath, error, meta = file_handler.save_file(file)

        if not success:
            return jsonify({"error": error}), 400

        file_info = file_handler.get_file_info(filepath)

        # file_id: for zip uploads, filepath will be inside uploads/<zipstem>/...
        # We want a stable id, so use the top-level folder/file stem.
        p = Path(filepath).resolve()
        root = file_handler.upload_folder.resolve()

        try:
            rel = p.relative_to(root)
            # If extracted zip: uploads/<file_id>/<video>
            # If direct video: uploads/<video>
            if len(rel.parts) >= 2:
                file_id = rel.parts[0]
            else:
                file_id = p.stem
        except Exception:
            # Fallback
            file_id = p.stem

        processing_status[file_id] = {
            "status": "uploaded",
            "filepath": filepath,         # points to main meeting video
            "file_info": file_info,
            "meta": meta or {},
        }

        return jsonify(
            {
                "message": "File uploaded successfully",
                "file_id": file_id,
                "file_info": file_info,
                "meta": meta or {},
            }
        ), 200

    except Exception as e:
        return jsonify({"error": "Upload failed", "details": str(e)}), 500


@api.route("/process/<file_id>", methods=["POST"])
def process_video(file_id: str):
    """
    Start processing a video
    Optional query params:
    - save_output: boolean (default: true)
    """
    try:
        if file_id not in processing_status:
            return jsonify({"error": "File not found"}), 404

        filepath = processing_status[file_id]["filepath"]
        save_output = request.args.get("save_output", "true").lower() == "true"

        processing_status[file_id]["status"] = "processing"

        processor = MeetingProcessor(filepath)
        results = processor.process_all(save_output=save_output)

        processing_status[file_id]["status"] = "completed"
        processing_status[file_id]["results"] = results

        return jsonify({"message": "Processing completed", "file_id": file_id, "results": results}), 200

    except Exception as e:
        processing_status[file_id]["status"] = "failed"
        processing_status[file_id]["error"] = str(e)

        return jsonify(
            {
                "error": "Processing failed",
                "details": str(e),
                "traceback": traceback.format_exc(),
            }
        ), 500


@api.route("/status/<file_id>", methods=["GET"])
def get_status(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404
    return jsonify(processing_status[file_id]), 200


@api.route("/results/<file_id>", methods=["GET"])
def get_results(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    status = processing_status[file_id]
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed", "current_status": status["status"]}), 400

    return jsonify(status.get("results", {})), 200


@api.route("/results/<file_id>/summary", methods=["GET"])
def get_summary(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    status = processing_status[file_id]
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed", "current_status": status["status"]}), 400

    results = status.get("results", {})
    return jsonify(
        {
            "summary": results.get("insights", {}).get("summary", {}),
            "action_items": results.get("insights", {}).get("action_items", []),
            "metrics": results.get("insights", {}).get("meeting_metrics", {}),
            "overall_mood": results.get("sentiment", {}).get("overall_mood", {}),
        }
    ), 200


@api.route("/results/<file_id>/participants", methods=["GET"])
def get_participants(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    status = processing_status[file_id]
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed", "current_status": status["status"]}), 400

    results = status.get("results", {})
    return jsonify(
        {
            "participants": results.get("participants", {}),
            "count": results.get("participants", {}).get("count", 0),
            "names": results.get("participants", {}).get("detected_names", []),
        }
    ), 200


@api.route("/results/<file_id>/timeline", methods=["GET"])
def get_timeline(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    status = processing_status[file_id]
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed", "current_status": status["status"]}), 400

    results = status.get("results", {})
    return jsonify(
        {
            "timeline": results.get("timeline", {}),
            "mood_changes": results.get("timeline", {}).get("mood_changes", []),
            "key_moments": results.get("timeline", {}).get("key_moments", []),
        }
    ), 200


@api.route("/results/<file_id>/sentiment", methods=["GET"])
def get_sentiment(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    status = processing_status[file_id]
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed", "current_status": status["status"]}), 400

    results = status.get("results", {})
    return jsonify(results.get("sentiment", {})), 200


@api.route("/results/<file_id>/transcript", methods=["GET"])
def get_transcript(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    status = processing_status[file_id]
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed", "current_status": status["status"]}), 400

    results = status.get("results", {})
    return jsonify(results.get("transcription", {})), 200


@api.route("/results/<file_id>/topics", methods=["GET"])
def get_topics(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    status = processing_status[file_id]
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed", "current_status": status["status"]}), 400

    results = status.get("results", {})
    return jsonify(results.get("topics", {})), 200


@api.route("/results/<file_id>/download", methods=["GET"])
def download_results(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    status = processing_status[file_id]
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed", "current_status": status["status"]}), 400

    results = status.get("results", {})
    output_path = results.get("output_path")

    if output_path and os.path.exists(output_path):
        return send_file(
            output_path,
            mimetype="application/json",
            as_attachment=True,
            download_name=f"{file_id}_analysis.json",
        )

    return jsonify(results), 200


@api.route("/delete/<file_id>", methods=["DELETE"])
def delete_video(file_id: str):
    if file_id not in processing_status:
        return jsonify({"error": "File not found"}), 404

    try:
        filepath = processing_status[file_id]["filepath"]
        file_handler.delete_file(filepath)
        del processing_status[file_id]

        return jsonify({"message": "File and results deleted successfully", "file_id": file_id}), 200

    except Exception as e:
        return jsonify({"error": "Deletion failed", "details": str(e)}), 500


@api.route("/list", methods=["GET"])
def list_files():
    files = []
    for file_id, status in processing_status.items():
        files.append(
            {"file_id": file_id, "status": status["status"], "file_info": status.get("file_info", {})}
        )

    return jsonify({"count": len(files), "files": files}), 200


@api.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@api.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "details": str(error)}), 500
