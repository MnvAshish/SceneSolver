import sys
import os
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from collections import Counter
from ultralytics import YOLO
from typing import Tuple, List, Dict, Any

# --- Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import CLIPCrimeClassifier
from scripts.constants import ( # Import constants from the constants.py file
    FRAME_INTERVAL, CAPTION_MAX_LENGTH, CAPTION_NUM_BEAMS,
    CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD,
    IDX_TO_LABEL, NUM_CLASSES, BINARY_IDX_TO_LABEL, NUM_BINARY_CLASSES,
    CRIME_OBJECTS, OBJ_CONF_THRESHOLD, VIDEO_DOM_THRESHOLD,
    BINARY_NORMAL_CONF_THRESHOLD, MULTI_CLASS_CRIME_CONF_THRESHOLD, SINGLE_FRAME_ALERT_THRESHOLD
)

# --- Core Processing Functions (Now supporting batch inputs) ---

def classify_binary_frame_batch(inputs_batch: torch.Tensor, binary_classifier_model: CLIPCrimeClassifier, device: torch.device) -> Tuple[List[str], List[float]]:
    """Classifies a batch of images as 'Crime' or 'Normal'."""
    print("DEBUG: Starting binary classification batch inference.")
    with torch.no_grad():
        logits = binary_classifier_model(inputs_batch)
        probs = torch.softmax(logits, dim=1)
        confidences, indices = torch.max(probs, dim=1)
    
    labels = [BINARY_IDX_TO_LABEL[idx.item()] for idx in indices]
    confs = confidences.tolist()
    print("DEBUG: Finished binary classification batch inference.")
    return labels, confs

def classify_frame_batch(inputs_batch: torch.Tensor, classifier_model: CLIPCrimeClassifier, device: torch.device) -> Tuple[List[str], List[float]]:
    """
    Classifies a batch of images into specific crime types.
    Includes custom logic: if the top prediction is 'Shooting' and the second best is 'Theft',
    the output is changed to 'Theft'.
    """
    print("DEBUG: Starting multi-class classification batch inference.")
    with torch.no_grad():
        logits = classifier_model(inputs_batch)
        probs = torch.softmax(logits, dim=1) # Get probabilities

        # Get top 2 predictions for each item in the batch
        # torch.topk returns (values, indices)
        # Ensure k is not greater than the number of classes
        k_val = min(2, NUM_CLASSES) 
        top_probs, top_indices = torch.topk(probs, k=k_val, dim=1)
    
    final_labels = []
    final_confs = []

    for i in range(inputs_batch.shape[0]): # Iterate through each item in the batch
        top1_idx = top_indices[i, 0].item()
        top1_prob = top_probs[i, 0].item()
        top1_label = IDX_TO_LABEL[top1_idx]

        # Default to top-1 prediction
        current_label = top1_label
        current_conf = top1_prob

        # Apply the specific logic: if top1 is "Shooting" AND top2 is "Theft"
        if k_val > 1: # Ensure there is a second prediction available
            top2_idx = top_indices[i, 1].item()
            top2_prob = top_probs[i, 1].item()
            top2_label = IDX_TO_LABEL[top2_idx]

            # Check if top1 is "Shooting" and top2 is "Theft"
            if top1_label == "Shooting" and top2_label == "Theft":
                # Override to "Theft" and use its confidence
                current_label = "Theft"
                current_conf = top2_prob 
                print(f"DEBUG: Applied Shooting/Theft mitigation: Changed from '{top1_label}' ({top1_prob:.2f}) to '{current_label}' ({current_conf:.2f}) for a frame.")
            # You could add an optional check here if top2_prob needs to be above a certain threshold
            # e.g., `and top2_prob > 0.40` to only switch if "Theft" is somewhat confident.
            # For now, it switches if "Theft" is the second best.

        final_labels.append(current_label)
        final_confs.append(current_conf)

    print("DEBUG: Finished multi-class classification batch inference with custom logic.")
    return final_labels, final_confs

def caption_frame_batch(images_batch: List[Image.Image], blip_processor: BlipProcessor, blip_model: BlipForConditionalGeneration, device: torch.device) -> List[str]:
    """Generates captions for a batch of images."""
    print(f"DEBUG: Starting BLIP captioning for a batch of {len(images_batch)} images.")
    
    print("DEBUG: BLIP: Preparing inputs with blip_processor.")
    inputs = blip_processor(images=images_batch, return_tensors="pt").to(device)
    
    print("DEBUG: BLIP: Calling blip_model.generate().")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=CAPTION_MAX_LENGTH, num_beams=CAPTION_NUM_BEAMS, early_stopping=True)
    
    print("DEBUG: BLIP: Calling blip_processor.batch_decode().")
    captions = blip_processor.batch_decode(out, skip_special_tokens=True)
    
    print("DEBUG: Finished BLIP captioning batch inference.")
    return captions

def detect_objects_batch(frames_batch: List[Any], yolo_model: YOLO) -> List[List[str]]:
    """Detects specified objects in a batch of video frames."""
    print("DEBUG: Starting YOLO object detection batch inference.")
    results_batch = yolo_model(frames_batch, verbose=False, conf=OBJ_CONF_THRESHOLD)
    
    all_detected_objs = []
    for results in results_batch:
        detected_objs_for_frame = []
        if results.boxes:
            for box in results.boxes:
                label = yolo_model.model.names[int(box.cls)]
                if label in CRIME_OBJECTS:
                    detected_objs_for_frame.append(label)
        all_detected_objs.append(detected_objs_for_frame)
    print("DEBUG: Finished YOLO object detection batch inference.")
    return all_detected_objs

def analyze_frame_batch(video_path: str, classifier_model: CLIPCrimeClassifier, binary_classifier_model: CLIPCrimeClassifier,
                  blip_processor: BlipProcessor, blip_model: BlipForConditionalGeneration, yolo_model: YOLO,
                  clip_transform: T.Compose, device: torch.device) -> Dict[str, Any]:
    """
    Processes a video by analyzing frames at a set interval using batch processing.
    """
    print(f"DEBUG: process_video function called for video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}", file=sys.stderr)
        raise RuntimeError(f"Error opening video file: {video_path}")
    print("DEBUG: cv2.VideoCapture opened successfully.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"DEBUG: Video FPS: {fps}, Total Frames: {frame_count_total}")
    
    results = {
        "frame_labels": [],
        "frame_confs": [],
        "detected_objects": [],
        "captions": [],
        "video_fps": fps
    }
    frame_idx = 0

    BATCH_SIZE = 8 # Adjust based on your GPU memory and desired speedup
    
    pil_images_batch = []
    cv_frames_batch = []
    frame_indices_batch = []

    while True: # Loop until video ends or error
        ret, frame = cap.read()
        if not ret:
            print(f"DEBUG: End of video stream or frame read error at frame_idx {frame_idx}.")
            break

        if frame_idx % FRAME_INTERVAL == 0:
            print(f"DEBUG: Collecting frame {frame_idx} for batch processing.")
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_images_batch.append(pil_img)
            cv_frames_batch.append(frame)
            frame_indices_batch.append(frame_idx)

            if len(pil_images_batch) >= BATCH_SIZE:
                print(f"DEBUG: Batch full ({len(pil_images_batch)} frames). Processing batch from index {frame_indices_batch[0]} to {frame_indices_batch[-1]}.")
                
                # --- Prepare batched inputs for CLIP models ---
                print("DEBUG: Preparing CLIP inputs batch.")
                clip_inputs_batch = torch.stack([clip_transform(img) for img in pil_images_batch]).to(device)
                
                # --- Perform Batch Inferences ---
                batch_binary_labels, batch_binary_confs = classify_binary_frame_batch(clip_inputs_batch, binary_classifier_model, device)
                
                frames_for_detailed_analysis_indices_in_batch = []
                pil_images_for_detailed_analysis = []

                for i, (binary_label, binary_conf) in enumerate(zip(batch_binary_labels, batch_binary_confs)):
                    run_detailed_analysis = False
                    if binary_label == "Crime":
                        run_detailed_analysis = True
                    elif binary_label == "Normal" and binary_conf <= BINARY_NORMAL_CONF_THRESHOLD:
                        run_detailed_analysis = True
                    
                    if run_detailed_analysis:
                        frames_for_detailed_analysis_indices_in_batch.append(i)
                        pil_images_for_detailed_analysis.append(pil_images_batch[i])

                batch_crime_labels_detailed = []
                batch_crime_confs_detailed = []
                batch_captions_detailed = []

                if frames_for_detailed_analysis_indices_in_batch:
                    print(f"DEBUG: {len(frames_for_detailed_analysis_indices_in_batch)} frames require detailed analysis.")
                    clip_inputs_detailed_batch = torch.stack([clip_transform(pil_images_batch[i]) for i in frames_for_detailed_analysis_indices_in_batch]).to(device)
                    
                    batch_crime_labels_detailed, batch_crime_confs_detailed = classify_frame_batch(clip_inputs_detailed_batch, classifier_model, device)
                    batch_captions_detailed = caption_frame_batch(pil_images_for_detailed_analysis, blip_processor, blip_model, device)
                else:
                    print("DEBUG: No frames in this batch require detailed analysis.")

                batch_detected_objects_lists = detect_objects_batch(cv_frames_batch, yolo_model)

                # --- Unpack and Store Results ---
                detailed_analysis_results_ptr = 0
                for i in range(len(pil_images_batch)):
                    current_frame_label = "Normal Activity"
                    current_frame_conf = batch_binary_confs[i]
                    current_frame_caption = "Normal activity observed."
                    
                    if i in frames_for_detailed_analysis_indices_in_batch:
                        crime_label = batch_crime_labels_detailed[detailed_analysis_results_ptr]
                        crime_conf = batch_crime_confs_detailed[detailed_analysis_results_ptr]
                        caption = batch_captions_detailed[detailed_analysis_results_ptr]

                        if crime_conf > MULTI_CLASS_CRIME_CONF_THRESHOLD:
                            current_frame_label = crime_label
                            current_frame_conf = crime_conf
                            current_frame_caption = caption
                            if crime_conf >= SINGLE_FRAME_ALERT_THRESHOLD:
                                print(f"    🚨 HIGH CONFIDENCE ALERT: Detected '{crime_label}' with {crime_conf:.2f} confidence in frame {frame_indices_batch[i]}.")
                        else:
                            print(f"    - Multi-class confidence ({crime_conf:.2f}) not greater than threshold ({MULTI_CLASS_CRIME_CONF_THRESHOLD:.2f}) for frame {frame_indices_batch[i]}. Treating as Normal Activity.")
                        
                        detailed_analysis_results_ptr += 1
                    else:
                        print(f"    - Binary model confident in 'Normal' result (conf > {BINARY_NORMAL_CONF_THRESHOLD:.2f}) for frame {frame_indices_batch[i]}. Skipping detailed analysis.")

                    results["frame_labels"].append(current_frame_label)
                    results["frame_confs"].append(current_frame_conf)
                    results["captions"].append(current_frame_caption)
                    results["detected_objects"].extend(batch_detected_objects_lists[i])

                # Clear the batch for the next set of frames
                pil_images_batch = []
                cv_frames_batch = []
                frame_indices_batch = []
                print("DEBUG: Batch cleared for next processing.")

        frame_idx += 1

    # Process any remaining frames in the batch after the loop finishes
    if pil_images_batch:
        print(f"DEBUG: Processing final remaining batch of {len(pil_images_batch)} frames (indices: {frame_indices_batch[0]} to {frame_indices_batch[-1]})...")
        
        print("DEBUG: Preparing CLIP inputs for final batch.")
        clip_inputs_batch = torch.stack([clip_transform(img) for img in pil_images_batch]).to(device)
        
        batch_binary_labels, batch_binary_confs = classify_binary_frame_batch(clip_inputs_batch, binary_classifier_model, device)
        
        frames_for_detailed_analysis_indices_in_batch = []
        pil_images_for_detailed_analysis = []

        for i, (binary_label, binary_conf) in enumerate(zip(batch_binary_labels, batch_binary_confs)):
            run_detailed_analysis = False
            if binary_label == "Crime":
                run_detailed_analysis = True
            elif binary_label == "Normal" and binary_conf <= BINARY_NORMAL_CONF_THRESHOLD:
                run_detailed_analysis = True
            
            if run_detailed_analysis:
                frames_for_detailed_analysis_indices_in_batch.append(i)
                pil_images_for_detailed_analysis.append(pil_images_batch[i])

        batch_crime_labels_detailed = []
        batch_crime_confs_detailed = []
        batch_captions_detailed = []

        if frames_for_detailed_analysis_indices_in_batch:
            print(f"DEBUG: {len(frames_for_detailed_analysis_indices_in_batch)} frames in final batch require detailed analysis.")
            clip_inputs_detailed_batch = torch.stack([clip_transform(pil_images_batch[i]) for i in frames_for_detailed_analysis_indices_in_batch]).to(device)
            batch_crime_labels_detailed, batch_crime_confs_detailed = classify_frame_batch(clip_inputs_detailed_batch, classifier_model, device)
            batch_captions_detailed = caption_frame_batch(pil_images_for_detailed_analysis, blip_processor, blip_model, device)
        else:
            print("DEBUG: No frames in final batch require detailed analysis.")

        batch_detected_objects_lists = detect_objects_batch(cv_frames_batch, yolo_model)

        detailed_analysis_results_ptr = 0
        for i in range(len(pil_images_batch)):
            current_frame_label = "Normal Activity"
            current_frame_conf = batch_binary_confs[i]
            current_frame_caption = "Normal activity observed."
            
            if i in frames_for_detailed_analysis_indices_in_batch:
                crime_label = batch_crime_labels_detailed[detailed_analysis_results_ptr]
                crime_conf = batch_crime_confs_detailed[detailed_analysis_results_ptr]
                caption = batch_captions_detailed[detailed_analysis_results_ptr]

                if crime_conf > MULTI_CLASS_CRIME_CONF_THRESHOLD:
                    current_frame_label = crime_label
                    current_frame_conf = crime_conf
                    current_frame_caption = caption
                    if crime_conf >= SINGLE_FRAME_ALERT_THRESHOLD:
                        print(f"    🚨 HIGH CONFIDENCE ALERT: Detected '{crime_label}' with {crime_conf:.2f} confidence in frame {frame_indices_batch[i]}.")
                else:
                    print(f"    - Multi-class confidence ({crime_conf:.2f}) not greater than threshold ({MULTI_CLASS_CRIME_CONF_THRESHOLD:.2f}) for frame {frame_indices_batch[i]}. Treating as Normal Activity.")
                
                detailed_analysis_results_ptr += 1
            else:
                print(f"    - Binary model confident in 'Normal' result (conf > {BINARY_NORMAL_CONF_THRESHOLD:.2f}) for frame {frame_indices_batch[i]}. Skipping detailed analysis.")

            results["frame_labels"].append(current_frame_label)
            results["frame_confs"].append(current_frame_conf)
            results["captions"].append(current_frame_caption)
            results["detected_objects"].extend(batch_detected_objects_lists[i])

    cap.release()
    return results

def aggregate_labels(labels: list, confs: list) -> Tuple[str, float]:
    """Aggregates frame-level labels into an overall video conclusion."""
    if not labels:
        return "No Activity Detected", 0.0

    all_label_counts = Counter(labels)
    
    most_common_label, count = all_label_counts.most_common(1)[0]
    
    dominance = count / len(labels)
    
    if most_common_label == "Normal Activity":
        if dominance >= VIDEO_DOM_THRESHOLD:
            return "Normal Activity Verified", dominance
        else:
            specific_crime_labels = [lbl for lbl in labels if lbl != "Normal Activity"]
            if specific_crime_labels:
                most_common_crime, crime_count = Counter(specific_crime_labels).most_common(1)[0]
                return f"Mixed Incident (predominantly Normal, with {most_common_crime})", dominance
            else:
                return "Normal Activity Verified", dominance # Should not happen if specific_crime_labels is empty
    else: # Most common label is a crime
        if dominance >= VIDEO_DOM_THRESHOLD:
            return most_common_label, dominance
        else:
            return f"Mixed Incident (featuring {most_common_label})", dominance

def summarize_captions(caps: list, summarizer: pipeline, overall_crime_class: str) -> str:
    """Summarizes a list of captions, with special handling for normal videos."""
    if "Normal Activity" in overall_crime_class or "Normal" in overall_crime_class: # Check for "Normal" in general
        return "The video footage was analyzed and determined to show routine activities with no significant crime-related events detected."

    text = " ".join(set(caps)) # Use set to avoid summarizing duplicate captions
    if not text or text == "Normal activity observed.":
        return "No specific details could be extracted from the scene."
    try:
        summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}. Returning raw text.", file=sys.stderr)
        return (text[:750] + "...") if len(text) > 750 else text
