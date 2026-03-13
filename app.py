import os
import sys
import time
import traceback
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from functools import wraps
from datetime import timedelta
import torch
import torchvision.transforms as T
from transformers import pipeline
from collections import Counter

# --- Setup Project Path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import Custom Modules ---
from backend.load_models import load_all
from video_crime_analyzer import process_video, aggregate_labels, summarize_captions
from scripts.constants import CLIP_MEAN, CLIP_STD

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'a_default_fallback_key_for_dev')
app.permanent_session_lifetime = timedelta(hours=1)
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'static', 'uploads')
app.config['CLIPS_FOLDER'] = os.path.join(PROJECT_ROOT, 'static', 'uploads', 'clips')
os.makedirs(os.path.join(PROJECT_ROOT, 'static', 'uploads', 'clips'), exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Database Connection ---
try:
    MONGO_URI = os.environ.get("MONGO_URI")
    if not MONGO_URI:
        print("CRITICAL ERROR: MONGO_URI environment variable not set.", file=sys.stderr)
        sys.exit(1)
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000
    )
    db = client['scene_solver_db']
    users = db['users']
    analysis_history = db['analysis_history']
    print("✅ MongoDB Atlas connected successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to MongoDB Atlas: {e}", file=sys.stderr)
    sys.exit(1)

# --- Global Model Variables & Immediate Loading ---
print("--- Initializing SceneSolver Models ---")
preferred_device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    GLOBAL_MODELS = load_all(device=preferred_device)
    clip_transform_global = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    print("✅ All Base Models Loaded Successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load models: {e}", file=sys.stderr)
    GLOBAL_MODELS = {}
    clip_transform_global = None

SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"
summarizer_pipeline_global = None

def load_summarizer_if_needed():
    global summarizer_pipeline_global
    if summarizer_pipeline_global is None:
        print("INFO: Loading summarizer model (lazy load)...")
        device_obj = GLOBAL_MODELS.get("device", torch.device("cpu"))
        device_id = 0 if device_obj.type == "cuda" else -1
        summarizer_pipeline_global = pipeline("summarization", model=SUMMARIZER_MODEL_NAME, device=device_id)
        if device_obj.type == 'cpu':
            print("INFO: Applying dynamic quantization to Summarizer model...")
            summarizer_pipeline_global.model = torch.quantization.quantize_dynamic(
                summarizer_pipeline_global.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("✅ Summarizer model dynamically quantized.")
        print("✅ Summarizer loaded.")
    return summarizer_pipeline_global

# --- Auth Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---
@app.route('/')
def home():
    return redirect(url_for('signin'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return redirect(url_for('register'))
        if users.find_one({'username': username}):
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password)
        users.insert_one({
            'username': username,
            'password': hashed_password,
            'created_at': datetime.datetime.now()
        })
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('signin'))
    return render_template('register.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        user = users.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['user_id'] = str(user['_id'])
            session.permanent = True
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('frontpage'))
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('signin'))
    return render_template('signin.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('signin'))

@app.route('/frontpage')
@login_required
def frontpage():
    return render_template('frontpage.html', username=session.get('username'))

@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        stream_url = request.form.get('stream_url', '').strip()
        print(f"DEBUG: stream_url='{stream_url}', video_file={request.files.get('video_file')}")
        video_file = request.files.get('video_file')

        video_source = None
        filename = None
        max_frames_to_process = None
        is_stream = False

        # --- Determine source: stream takes priority ---
        if stream_url:
            video_source = 0 if stream_url == '0' else stream_url
            filename = f"stream_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            max_frames_to_process = 900  # ~30 seconds at 30fps
            is_stream = True

        elif video_file and video_file.filename != '':
            filename = os.path.basename(video_file.filename)
            video_source = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_source)
            is_stream = False

        else:
            flash('Please upload a video file or enter a stream URL.', 'error')
            return redirect(request.url)

        try:
            start_time = time.time()
            print(f"\n--- Starting Analysis for: {filename} ---")

            if "classifier_model" not in GLOBAL_MODELS:
                raise KeyError("Models were not loaded properly. Check terminal for startup errors.")

            analysis_result = process_video(
                video_path=video_source,
                classifier_model=GLOBAL_MODELS["classifier_model"],
                binary_classifier_model=GLOBAL_MODELS["binary_model"],
                blip_processor=GLOBAL_MODELS["blip_processor"],
                blip_model=GLOBAL_MODELS["blip_model"],
                yolo_model=GLOBAL_MODELS["yolo_model"],
                clip_transform=clip_transform_global,
                device=GLOBAL_MODELS["device"],
                max_frames=max_frames_to_process,
                clips_output_dir=app.config['CLIPS_FOLDER']
            )

            print(f"DEBUG: analysis_result keys={analysis_result.keys()}, captions={analysis_result.get('captions')}, labels={analysis_result.get('frame_labels')}")
            if not analysis_result or not analysis_result.get("captions"):
                flash("Video analysis failed. The video might be too short or corrupted.", 'error')
                if not is_stream and isinstance(video_source, str) and os.path.exists(video_source):
                    try:
                        os.remove(video_source)
                    except PermissionError:
                        pass
                return redirect(url_for('index'))

            video_crime_class, crime_dominance = aggregate_labels(
                analysis_result["frame_labels"],
                analysis_result["frame_confs"]
            )

            summarizer = load_summarizer_if_needed()

            # --- FIXED: correct signature includes detected_objects ---
            video_summary = summarize_captions(
                analysis_result["captions"],
                analysis_result["detected_objects"],
                summarizer,
                video_crime_class
            )

            # detected_objects is list of lists — flatten first
            flat_objects = [obj for sublist in analysis_result["detected_objects"] for obj in (sublist if isinstance(sublist, list) else [sublist])]
            object_counts = Counter(flat_objects)
            top_objects_display = [
                f"{label} (seen in {count} frames)"
                for label, count in object_counts.most_common(5)
            ]
            if not top_objects_display:
                top_objects_display.append("No notable objects detected.")

            total_duration = time.time() - start_time
            print(f"--- Analysis Complete for {filename} in {total_duration:.2f}s ---")

            # --- Safe video URL: no file saved for streams ---
            safe_video_url = "#" if is_stream else url_for('uploaded_file', filename=filename)

            session['analysis_results'] = {
                'video_file_name': filename,
                'overall_crime': video_crime_class,
                'confidence_score': crime_dominance,
                'detected_objects': top_objects_display,
                'summary': video_summary,
                'analysis_duration': f"{total_duration:.2f}",
                'video_url': safe_video_url,
                'is_stream': is_stream,
                'crime_clips': analysis_result.get('crime_clips', [])[:3],  # max 3 clips in session
            }

            analysis_history.insert_one({
                'username': session.get('username'),
                'filename': filename,
                'upload_time': datetime.datetime.now(),
                'crime_type': video_crime_class,
                'confidence_score': crime_dominance,
                'detected_objects': list(object_counts.keys()),
                'summary': video_summary,
                'video_url': safe_video_url
            })
            flash('Analysis complete and saved to history.', 'success')
            return redirect(url_for('result'))

        except Exception as e:
            flash(f"An error occurred during analysis: {e}", 'error')
            print(f"ERROR during video analysis: {e}", file=sys.stderr)
            traceback.print_exc()
            if not is_stream and isinstance(video_source, str) and os.path.exists(video_source):
                try:
                    os.remove(video_source)
                except PermissionError:
                    pass
            return redirect(url_for('index'))

    return render_template('index.html', username=session.get('username'))

@app.route('/result')
@login_required
def result():
    analysis_data = session.pop('analysis_results', None)
    if not analysis_data:
        flash('No analysis results found. Please upload a video first.', 'error')
        return redirect(url_for('index'))
    return render_template('result.html', username=session.get('username'), **analysis_data)

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/clips/<filename>')
@login_required
def serve_clip(filename):
    return send_from_directory(app.config['CLIPS_FOLDER'], filename)

@app.route('/history')
@login_required
def history():
    username = session.get('username')
    try:
        user_history = list(analysis_history.find({'username': username}).sort('upload_time', -1))
        return render_template('history.html', username=username, history=user_history)
    except Exception as e:
        print(f"ERROR retrieving history: {e}", file=sys.stderr)
        flash('Could not retrieve analysis history.', 'error')
        return render_template('history.html', username=username, history=[])

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        if not feedback_text:
            flash('Feedback cannot be empty.', 'error')
            return redirect(url_for('feedback'))
        db['feedback'].insert_one({
            'username': session.get('username'),
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'feedback_text': feedback_text,
            'submitted_at': datetime.datetime.now()
        })
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('feedback'))
    return render_template('feedback.html', username=session.get('username'))

# --- Main Execution Block ---
if __name__ == '__main__':
    app.run(debug=True)