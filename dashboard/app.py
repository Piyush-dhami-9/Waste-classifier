"""
Smart Littering Detection - Web Dashboard
"""

from flask import Flask, render_template, jsonify, send_from_directory
import os
import csv
from datetime import datetime

# Paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_FOLDER = os.path.join(_PROJECT_ROOT, "evidence")
IMAGES_FOLDER = os.path.join(EVIDENCE_FOLDER, "images")
VIDEOS_FOLDER = os.path.join(EVIDENCE_FOLDER, "videos")
LOGS_FILE = os.path.join(EVIDENCE_FOLDER, "logs", "events.csv")

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))


@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/incidents')
def incidents_page():
    """Incidents list page."""
    return render_template('incidents.html')


@app.route('/api/incidents')
def get_incidents():
    """Get all littering incidents."""
    incidents = []
    
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('image_path'):
                        row['image_path'] = os.path.basename(row['image_path'])
                    if row.get('video_path'):
                        row['video_path'] = os.path.basename(row['video_path'])
                    incidents.append(row)
        except Exception as e:
            print(f"Error reading logs: {e}")
    
    if os.path.exists(IMAGES_FOLDER):
        for filename in os.listdir(IMAGES_FOLDER):
            if filename.endswith(('.jpg', '.png')):
                try:
                    parts = filename.replace('litter_', '').replace('.jpg', '').replace('.png', '')
                    timestamp = parts.replace('_', ' ').replace('-', ':', 2)
                    
                    exists = any(i.get('image_path', '').endswith(filename) for i in incidents)
                    
                    if not exists:
                        incidents.append({
                            'timestamp': timestamp,
                            'event_type': 'LITTERING',
                            'image_path': filename,
                            'video_path': '',
                            'garbage_class': 'Unknown',
                            'garbage_confidence': '0.0',
                            'person_confidence': '0.0'
                        })
                except Exception:
                    pass
    
    incidents.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify(incidents)


@app.route('/api/stats')
def get_stats():
    """Get statistics."""
    incidents = []
    
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, 'r') as f:
                reader = csv.DictReader(f)
                incidents = list(reader)
        except Exception:
            pass
    
    image_count = 0
    if os.path.exists(IMAGES_FOLDER):
        image_count = len([f for f in os.listdir(IMAGES_FOLDER) if f.endswith(('.jpg', '.png'))])
    
    video_count = 0
    if os.path.exists(VIDEOS_FOLDER):
        video_count = len([f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi'))])
    
    return jsonify({
        'total_incidents': max(len(incidents), image_count),
        'total_images': image_count,
        'total_videos': video_count,
        'today_incidents': sum(1 for i in incidents if datetime.now().strftime('%Y-%m-%d') in i.get('timestamp', ''))
    })


@app.route('/evidence/images/<filename>')
def serve_image(filename):
    """Serve evidence images."""
    return send_from_directory(IMAGES_FOLDER, filename)


@app.route('/evidence/videos/<filename>')
def serve_video(filename):
    """Serve evidence videos."""
    return send_from_directory(VIDEOS_FOLDER, filename)


if __name__ == '__main__':
    print("=" * 50)
    print("🌐 Smart Littering Detection - Dashboard")
    print("=" * 50)
    print(f"📁 Evidence folder: {EVIDENCE_FOLDER}")
    print(f"🖼️  Images folder: {IMAGES_FOLDER}")
    print(f"🎥 Videos folder: {VIDEOS_FOLDER}")
    print("=" * 50)
    print("🚀 Starting server at http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
