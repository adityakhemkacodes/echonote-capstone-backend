from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from typing import Dict, Any
import json
import traceback

from .file_handler import FileHandler
from .processors import MeetingProcessor, process_meeting_video

# Create Blueprint
api = Blueprint('api', __name__)

# Initialize file handler
file_handler = FileHandler()

# Store processing status (in production, use a proper database or cache)
processing_status = {}


@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'meeting-analysis-api',
        'version': '1.0.0'
    }), 200


@api.route('/upload', methods=['POST'])
def upload_video():
    """
    Upload a video file for processing
    Returns: file_id and basic file information
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
        
        file = request.files['file']
        
        # Save file
        success, filepath, error = file_handler.save_file(file)
        
        if not success:
            return jsonify({'error': error}), 400
        
        # Get file info
        file_info = file_handler.get_file_info(filepath)
        file_id = Path(filepath).stem
        
        # Initialize processing status
        processing_status[file_id] = {
            'status': 'uploaded',
            'filepath': filepath,
            'file_info': file_info
        }
        
        return jsonify({
            'message': 'File uploaded successfully',
            'file_id': file_id,
            'file_info': file_info
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Upload failed',
            'details': str(e)
        }), 500


@api.route('/process/<file_id>', methods=['POST'])
def process_video(file_id: str):
    """
    Start processing a video
    Optional query params:
    - save_output: boolean (default: true)
    """
    try:
        if file_id not in processing_status:
            return jsonify({'error': 'File not found'}), 404
        
        filepath = processing_status[file_id]['filepath']
        save_output = request.args.get('save_output', 'true').lower() == 'true'
        
        # Update status
        processing_status[file_id]['status'] = 'processing'
        
        # Process video
        processor = MeetingProcessor(filepath)
        results = processor.process_all(save_output=save_output)
        
        # Update status with results
        processing_status[file_id]['status'] = 'completed'
        processing_status[file_id]['results'] = results
        
        return jsonify({
            'message': 'Processing completed',
            'file_id': file_id,
            'results': results
        }), 200
    
    except Exception as e:
        processing_status[file_id]['status'] = 'failed'
        processing_status[file_id]['error'] = str(e)
        
        return jsonify({
            'error': 'Processing failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api.route('/status/<file_id>', methods=['GET'])
def get_status(file_id: str):
    """Get processing status for a file"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify(processing_status[file_id]), 200


@api.route('/results/<file_id>', methods=['GET'])
def get_results(file_id: str):
    """Get full results for a processed video"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = processing_status[file_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    return jsonify(status.get('results', {})), 200


@api.route('/results/<file_id>/summary', methods=['GET'])
def get_summary(file_id: str):
    """Get only the summary and insights"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = processing_status[file_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    results = status.get('results', {})
    
    return jsonify({
        'summary': results.get('insights', {}).get('summary', {}),
        'action_items': results.get('insights', {}).get('action_items', []),
        'metrics': results.get('insights', {}).get('meeting_metrics', {}),
        'overall_mood': results.get('sentiment', {}).get('overall_mood', {})
    }), 200


@api.route('/results/<file_id>/participants', methods=['GET'])
def get_participants(file_id: str):
    """Get participant information"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = processing_status[file_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    results = status.get('results', {})
    
    return jsonify({
        'participants': results.get('participants', {}),
        'count': results.get('participants', {}).get('count', 0),
        'names': results.get('participants', {}).get('detected_names', [])
    }), 200


@api.route('/results/<file_id>/timeline', methods=['GET'])
def get_timeline(file_id: str):
    """Get timeline with mood changes"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = processing_status[file_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    results = status.get('results', {})
    
    return jsonify({
        'timeline': results.get('timeline', {}),
        'mood_changes': results.get('timeline', {}).get('mood_changes', []),
        'key_moments': results.get('timeline', {}).get('key_moments', [])
    }), 200


@api.route('/results/<file_id>/sentiment', methods=['GET'])
def get_sentiment(file_id: str):
    """Get sentiment analysis results"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = processing_status[file_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    results = status.get('results', {})
    
    return jsonify(results.get('sentiment', {})), 200


@api.route('/results/<file_id>/transcript', methods=['GET'])
def get_transcript(file_id: str):
    """Get transcription with speaker identification"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = processing_status[file_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    results = status.get('results', {})
    
    return jsonify(results.get('transcription', {})), 200


@api.route('/results/<file_id>/topics', methods=['GET'])
def get_topics(file_id: str):
    """Get topic segmentation"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = processing_status[file_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    results = status.get('results', {})
    
    return jsonify(results.get('topics', {})), 200


@api.route('/results/<file_id>/download', methods=['GET'])
def download_results(file_id: str):
    """Download results as JSON file"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    status = processing_status[file_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'current_status': status['status']
        }), 400
    
    results = status.get('results', {})
    output_path = results.get('output_path')
    
    if output_path and os.path.exists(output_path):
        return send_file(
            output_path,
            mimetype='application/json',
            as_attachment=True,
            download_name=f'{file_id}_analysis.json'
        )
    
    # If no saved file, return results as JSON
    return jsonify(results), 200


@api.route('/delete/<file_id>', methods=['DELETE'])
def delete_video(file_id: str):
    """Delete uploaded video and associated results"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    try:
        filepath = processing_status[file_id]['filepath']
        file_handler.delete_file(filepath)
        
        # Clean up processing status
        del processing_status[file_id]
        
        return jsonify({
            'message': 'File and results deleted successfully',
            'file_id': file_id
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Deletion failed',
            'details': str(e)
        }), 500


@api.route('/list', methods=['GET'])
def list_files():
    """List all uploaded/processed files"""
    files = []
    for file_id, status in processing_status.items():
        files.append({
            'file_id': file_id,
            'status': status['status'],
            'file_info': status.get('file_info', {})
        })
    
    return jsonify({
        'count': len(files),
        'files': files
    }), 200


# Error handlers
@api.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@api.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'details': str(error)
    }), 500