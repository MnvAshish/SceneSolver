import os
import requests
import zipfile

models = {
    "blip": "https://drive.google.com/file/d/1zEiuuSOYjOC3_KRLZielWzxtxuDoTVUI/view?usp=sharing",
    "yolo": "https://drive.google.com/file/d/1hrHIWkeR5YnqnP1LeL_EbU2JyTDrF8Pm/view?usp=drive_link",
    "weapon_classifier": "https://drive.google.com/file/d/1ZlVStd3TRymeB8ZXG8gAKOeaHg-U-o52/view?usp=drive_link",
    "binary_classifier": "https://drive.google.com/file/d/1yevkrwkJ-ZHw-sGm4bRNqXC0DmPnOAgm/view?usp=drive_link",
    "multi_classifier": "https://drive.google.com/file/d/1NwNSaoBODNYcCtGK5X1_6VnJBTQPNxwo/view?usp=drive_link"
}

def download_file_from_google_drive(file_id, dest_path):
    if os.path.exists(dest_path):
        print(f"✅ {dest_path} already exists.")
        return
    print(f"⬇️ Downloading {dest_path} from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

def download_and_extract_zip(file_id, output_dir):
    if os.path.exists(os.path.join(output_dir, "model.safetensors")):
        print(f"✅ BLIP model already exists.")
        return
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "blip_model.zip")
    download_file_from_google_drive(file_id, zip_path)
    print("📦 Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(zip_path)

def ensure_models_downloaded():
    base = "/tmp/models"
    os.makedirs(base, exist_ok=True)
    
    # BLIP
    download_and_extract_zip(models["blip"], os.path.join(base, "blip_finetuned_crime"))

    # Flat .pt files
    flat_models = ["yolo", "weapon_classifier", "binary_classifier", "multi_classifier"]
    for model_name in flat_models:
        out_path = os.path.join(base, f"{model_name}.pt")
        download_file_from_google_drive(models[model_name], out_path)

    return {
        "blip": os.path.join(base, "blip_finetuned_crime"),
        "yolo": os.path.join(base, "yolo.pt"),
        "weapon_classifier": os.path.join(base, "weapon_classifier.pt"),
        "binary_classifier": os.path.join(base, "binary_classifier.pt"),
        "multi_classifier": os.path.join(base, "multi_classifier.pt")
    }
