"""
app_demo.py — SceneSolver Demo Deployment
==========================================
Zero ML models. Zero GPU. Boots in ~5s on Render free tier.
Set env var PORT (Render sets this automatically).

Routes:
  /           → redirects to /demo
  /demo       → full result page with pre-computed fighting result
  /feedback   → optional feedback form (MongoDB optional)
"""

import os
from flask import Flask, render_template, redirect, url_for, request, flash, send_from_directory

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'demo-secret-key-change-me')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_FOLDER  = os.path.join(PROJECT_ROOT, 'static', 'demo')
os.makedirs(DEMO_FOLDER, exist_ok=True)

# ── Optional MongoDB for feedback (gracefully skipped if no MONGO_URI) ──────
feedback_collection = None
try:
    mongo_uri = os.environ.get('MONGO_URI')
    if mongo_uri:
        from pymongo import MongoClient
        _client = MongoClient(
            mongo_uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000
        )
        feedback_collection = _client['scene_solver_db']['feedback']
        print("✅ MongoDB connected (feedback only).")
    else:
        print("ℹ️  No MONGO_URI set — feedback will be disabled.")
except Exception as e:
    print(f"⚠️  MongoDB optional — skipping: {e}")

# ── Pre-computed demo result (real output from the model pipeline) ───────────
DEMO_DATA = {
    'video_file_name': 'demo_fighting_clip.mp4',
    'overall_crime':   'Fighting',
    'confidence_score': 1.0,
    'detected_objects': [
        'person (seen in 6 frames)',
        'person (seen in 4 frames)',
        'person (seen in 3 frames)',
    ],
    'summary': (
        'The footage captures a violent altercation between multiple individuals outside a '
        'commercial premises. One suspect delivers a powerful blow directly at an opponent '
        'near the storefront entrance. Five individuals are tracked across multiple frames '
        'with consistent IDs indicating sustained confrontation. Law enforcement response '
        'is advised. Incident classified as Fighting with high confidence.'
    ),
    'analysis_duration': '79.82',
    'video_url':  '/static/demo/demo_fighting.mp4',
    'is_stream':  False,
    'crime_clips': [{
        'filename':      'demo_clip_Fighting.mp4',
        'crime_label':   'Fighting',
        'trigger_frame': 0
    }],
}

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('signin.html')


@app.route('/demo')
def demo():
    """Public demo — no login, no models, instant."""
    return render_template('result.html', username='Guest', **DEMO_DATA)


@app.route('/static/demo/<filename>')
def serve_demo_file(filename):
    return send_from_directory(DEMO_FOLDER, filename)


# ── Stub routes so result.html / signin.html url_for() calls don't crash ─────

@app.route('/frontpage')
def frontpage():
    return redirect(url_for('demo'))

@app.route('/index')
def index():
    return redirect(url_for('demo'))

@app.route('/history')
def history():
    return redirect(url_for('demo'))

@app.route('/signin')
def signin():
    return redirect(url_for('demo'))

@app.route('/register')
def register():
    return redirect(url_for('demo'))

@app.route('/logout')
def logout():
    return redirect(url_for('demo'))

@app.route('/export-pdf')
def export_pdf():
    return redirect(url_for('demo'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('demo'))

@app.route('/clips/<filename>')
def serve_clip(filename):
    # Serve from demo folder — demo clips live there
    return send_from_directory(DEMO_FOLDER, filename)


# ── Feedback ─────────────────────────────────────────────────────────────────

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    import datetime
    if request.method == 'POST':
        feedback_text = request.form.get('feedback', '').strip()
        if not feedback_text:
            flash('Feedback cannot be empty.', 'error')
            return redirect(url_for('feedback'))
        if feedback_collection is not None:
            try:
                feedback_collection.insert_one({
                    'name':          request.form.get('name', 'Anonymous'),
                    'email':         request.form.get('email', ''),
                    'feedback_text': feedback_text,
                    'submitted_at':  datetime.datetime.now(),
                    'source':        'demo',
                })
                flash('Thank you for your feedback!', 'success')
            except Exception as e:
                print(f"Feedback save error: {e}")
                flash('Feedback received (storage unavailable).', 'success')
        else:
            flash('Thank you for your feedback!', 'success')
        return redirect(url_for('feedback'))
    return render_template('feedback.html', username='Guest')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

