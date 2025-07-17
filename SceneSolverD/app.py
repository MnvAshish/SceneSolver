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
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline, BitsAndBytesConfig
from ultralytics import YOLO # Ensure YOLO is imported globally here
from collections import Counter # Moved Counter import here to ensure it's always available
from model_loader import ensure_models_downloaded
from ultralytics import YOLO
from utils.video_crime_analyzer import video_crime_analyzer


model_paths = ensure_models_downloaded()
yolo_model = YOLO(model_paths["yolo"])


# --- Setup Project Path ---
# This ensures that imports from the 'scripts' directory work correctly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# --- Import Custom Modules and Constants ---
try:
    from models import CLIPCrimeClassifier
    from video_crime_analyzer import (
        process_video, aggregate_labels, summarize_captions
    )
    # Import constants from the new constants.py file
    from scripts.constants import (
        IDX_TO_LABEL, NUM_CLASSES, BINARY_IDX_TO_LABEL, NUM_BINARY_CLASSES,
        CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD, CAPTION_MAX_LENGTH
    )
    # Explicitly check for accelerate here, as it's a runtime dependency for BitsAndBytesConfig
    import accelerate
    print(f"INFO: accelerate version: {accelerate.__version__}")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import custom modules or required libraries: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed. For accelerate, run: pip install 'accelerate>=0.26.0'", file=sys.stderr)
    sys.exit(1)

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here_replace_this_in_production'
app.permanent_session_lifetime = timedelta(hours=1)
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'static', 'uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Database Connection ---
try:
    # It's good practice to use environment variables for credentials in production.
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://Yeshwanth:Yeshwanth%401505@cluster0.ybdqrez.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    client = MongoClient(MONGO_URI)
    db = client['scene_solver_db']
    users = db['users']
    analysis_history = db['analysis_history']
    print("✅ MongoDB Atlas connected successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to MongoDB Atlas: {e}", file=sys.stderr)
    sys.exit(1)

# --- Model Paths & Global Variables ---
# Centralized model paths using PROJECT_ROOT for portability.
CLASSIFIER_CKPT = os.path.join(PROJECT_ROOT, "models", "retrained_multi_class_classifier.pt")
BINARY_CLASSIFIER_CKPT = os.path.join(PROJECT_ROOT, "models", "retrained_binary_classifier.pt")
BLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "blip_finetuned_crime")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8n.pt")
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

# Global device is determined once.
GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Determined device: {GLOBAL_DEVICE}") # Added debug print

# Global model instances are initialized to None.
# They will be loaded *ONCE* by the load_models() function before the app starts.
classifier_model_global = None
binary_classifier_model_global = None
blip_processor_global = None
blip_model_global = None
yolo_model_global = None
summarizer_pipeline_global = None
clip_transform_global = None

def load_models():
    """
    Loads all machine learning models into global variables.
    This function is called only ONCE when the Flask application starts.
    Includes a fallback to CPU if GPU loading fails.
    """
    global classifier_model_global, binary_classifier_model_global, blip_processor_global
    global blip_model_global, yolo_model_global, summarizer_pipeline_global, clip_transform_global, GLOBAL_DEVICE
    
    print("\n--- Loading All Models ---")
    
    try:
        # Load binary classifier (Crime vs. Normal)
        print("INFO: Loading binary classifier...")
        binary_classifier_model_global = CLIPCrimeClassifier(num_classes=NUM_BINARY_CLASSES, freeze_clip=True, device=GLOBAL_DEVICE)
        binary_classifier_model_global.load_state_dict(torch.load(BINARY_CLASSIFIER_CKPT, map_location=GLOBAL_DEVICE), strict=False)
        binary_classifier_model_global.to(GLOBAL_DEVICE).eval()
        print("✅ Binary classifier loaded.")
        
        # Load multi-class classifier (Specific crime types)
        print("INFO: Loading multi-class classifier...")
        classifier_model_global = CLIPCrimeClassifier(num_classes=NUM_CLASSES, freeze_clip=True, device=GLOBAL_DEVICE)
        classifier_model_global.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=GLOBAL_DEVICE), strict=False)
        classifier_model_global.to(GLOBAL_DEVICE).eval()
        print("✅ Multi-class classifier loaded.")
        
        # Load BLIP model for captioning with 8-bit quantization
        print("INFO: Loading BLIP model with 8-bit quantization...")
        blip_processor_global = BlipProcessor.from_pretrained(BLIP_MODEL_PATH, use_fast=False)
        
        # Define quantization configuration
        # This will load the model in 8-bit if a GPU is available and bitsandbytes is installed
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_quant_type="nf4", # Optional: can try "fp4"
            bnb_4bit_compute_dtype=torch.float16, # Optional: for faster computation on some GPUs
            bnb_4bit_use_double_quant=True, # Optional: further quantization
        )

        # Load BLIP model with quantization if GPU is available, otherwise load normally to CPU
        if GLOBAL_DEVICE.type == 'cuda':
            blip_model_global = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_PATH, 
                quantization_config=bnb_config,
                torch_dtype=torch.float16 # Use float16 for potentially faster inference
            ).eval()
            # The .to(GLOBAL_DEVICE) is handled by quantization_config for GPU
        else:
            # If falling back to CPU, load without quantization as bitsandbytes is GPU-specific
            print("INFO: Loading BLIP model without 8-bit quantization (CPU fallback).")
            blip_model_global = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_PATH).to(GLOBAL_DEVICE).eval()

        print("✅ BLIP model loaded.")
        
        # Load YOLO model for object detection
        print("INFO: Loading YOLO model...")
        # Re-instantiate YOLO model here to ensure it's fresh for the current device
        yolo_model_global = YOLO(YOLO_MODEL_PATH)
        yolo_model_global.to(GLOBAL_DEVICE)
        print("✅ YOLO model loaded.")
        
        # Summarizer is loaded lazily (only when first needed)
        summarizer_pipeline_global = None 
        print("INFO: Summarizer will be loaded on first use (lazy loading).")
        
        # Setup the image transformation pipeline for CLIP models
        clip_transform_global = T.Compose([
            T.Resize((CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
        print("✅ CLIP transform configured.")
        
        print(f"\n--- All models loaded successfully on {GLOBAL_DEVICE}! ---\n") # Updated debug print
        
    except Exception as e:
        print(f"ERROR: Failed to load models on {GLOBAL_DEVICE}: {e}", file=sys.stderr)
        traceback.print_exc()
        # Fallback to CPU if GPU fails
        if GLOBAL_DEVICE.type == 'cuda':
            print("WARNING: Attempting to fall back to CPU for model loading...")
            GLOBAL_DEVICE = torch.device("cpu")
            load_models() # Retry loading on CPU
        else:
            print("CRITICAL: Models could not be loaded on CPU. The application cannot start.", file=sys.stderr)
            sys.exit(1)

def load_summarizer_if_needed():
    """Loads the summarizer model only on its first use to save memory."""
    global summarizer_pipeline_global
    if summarizer_pipeline_global is None:
        print("INFO: Loading summarizer model (lazy load)...")
        device_id = 0 if GLOBAL_DEVICE.type == "cuda" else -1
        summarizer_pipeline_global = pipeline("summarization", model=SUMMARIZER_MODEL, device=device_id)
        print("✅ Summarizer loaded.")
    return summarizer_pipeline_global

# --- User Authentication & Routes ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

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
        users.insert_one({'username': username, 'password': hashed_password, 'created_at': datetime.datetime.now()})
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
        if 'video_file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)
        
        video_file = request.files['video_file']
        if video_file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if video_file:
            filename = os.path.basename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)

            try:
                start_time = time.time()
                print(f"\n--- Starting Analysis for: {filename} ---")

                # Call the main processing function, passing all loaded models.
                # This ensures models are not reloaded for each request.
                analysis_result = process_video(
                    video_path=video_path, 
                    classifier_model=classifier_model_global, 
                    binary_classifier_model=binary_classifier_model_global,
                    blip_processor=blip_processor_global, 
                    blip_model=blip_model_global, 
                    yolo_model=yolo_model_global, 
                    clip_transform=clip_transform_global,      
                    device=GLOBAL_DEVICE             
                )

                if not analysis_result or not analysis_result.get("captions"):
                    flash("Video analysis failed. The video might be too short or corrupted.", 'error')
                    if os.path.exists(video_path): os.remove(video_path)
                    return redirect(url_for('index'))

                # Aggregate and summarize the results
                video_crime_class, crime_dominance = aggregate_labels(
                    analysis_result["frame_labels"], 
                    analysis_result["frame_confs"]
                )
                
                video_summary = summarize_captions(
                    analysis_result["captions"], 
                    load_summarizer_if_needed(), 
                    video_crime_class
                )

                # Format detected objects for display
                object_counts = Counter(analysis_result["detected_objects"])
                top_objects_display = [f"{label} (seen in {count} frames)" for label, count in object_counts.most_common(5)]
                if not top_objects_display:
                    top_objects_display.append("No notable objects detected.")

                total_duration = time.time() - start_time
                print(f"--- Analysis Complete for {filename} in {total_duration:.2f}s ---")

                # Store results in session to pass to the result page
                session['analysis_results'] = {
                    'video_file_name': filename,
                    'overall_crime': video_crime_class,
                    'confidence_score': crime_dominance,
                    'detected_objects': top_objects_display,
                    'summary': video_summary,
                    'analysis_duration': f"{total_duration:.2f}",
                    'video_url': url_for('uploaded_file', filename=filename)
                }

                # Save a record to MongoDB history
                analysis_history.insert_one({
                    'username': session.get('username'),
                    'filename': filename,
                    'upload_time': datetime.datetime.now(),
                    'crime_type': video_crime_class,
                    'confidence_score': crime_dominance,
                    'detected_objects': list(object_counts.keys()),
                    'summary': video_summary,
                    'video_url': url_for('uploaded_file', filename=filename)
                })
                flash('Analysis complete and saved to history.', 'success')

                return redirect(url_for('result'))
            
            except Exception as e:
                flash(f"An error occurred during analysis: {e}", 'error')
                print(f"ERROR during video analysis: {e}", file=sys.stderr)
                traceback.print_exc()
                if os.path.exists(video_path): os.remove(video_path)
                return redirect(url_for('index'))

    return render_template('index.html')


@app.route('/result')
@login_required
def result():
    """Displays the analysis results retrieved from the session."""
    analysis_data = session.pop('analysis_results', None)
    if not analysis_data:
        flash('No analysis results found. Please upload a video first.', 'error')
        return redirect(url_for('index'))
    return render_template('result.html', **analysis_data)


@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/history')
@login_required
def history():
    username = session.get('username')
    try:
        # Sort by upload_time descending (most recent first)
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
    # This is the crucial part: load_models() is called here, once,
    # before the web server starts accepting requests.
    load_models()
    
    # Use debug=False in a production environment
    app.run()
