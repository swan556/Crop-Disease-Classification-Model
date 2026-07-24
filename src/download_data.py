import os
import sys
import zipfile
import shutil
import requests  # pyrefly: ignore [missing-import]
from pathlib import Path
from tqdm import tqdm  # pyrefly: ignore [missing-import]

DATASET_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
TARGET_DIR = Path(__file__).resolve().parent.parent / "data" / "PlantVillage"
ZIP_PATH = Path(__file__).resolve().parent.parent / "data" / "plantvillage_master.zip"

def download_file(url: str, dest_path: Path):
    """Download a file with progress bar and resumption support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from: {url}")
    
    # Support resume if partial download exists
    headers = {}
    existing_size = 0
    if dest_path.exists():
        existing_size = dest_path.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
    
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    
    # If server doesn't support range requests, start fresh
    if response.status_code == 200:
        existing_size = 0
        mode = "wb"
    elif response.status_code == 206:
        mode = "ab"
    else:
        response.raise_for_status()
        mode = "wb"
    
    total_size = int(response.headers.get("content-length", 0)) + existing_size
    
    with open(dest_path, mode) as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        initial=existing_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Download complete -> {dest_path}")


def verify_zip(zip_path: Path) -> bool:
    """Verify that a zip file is valid and not corrupted."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # testzip() returns the name of first bad file, or None if all OK
            bad_file = zf.testzip()
            if bad_file is not None:
                print(f"Corrupted file in zip: {bad_file}")
                return False
        return True
    except (zipfile.BadZipFile, Exception) as e:
        print(f"Zip verification failed: {e}")
        return False


def extract_and_organize(zip_path: Path, target_dir: Path):
    """Extract zip file and organize color dataset images into target_dir."""
    # Verify zip integrity first
    if not verify_zip(zip_path):
        print("Zip file is corrupted or incomplete. Deleting and re-downloading...")
        zip_path.unlink(missing_ok=True)
        download_file(DATASET_URL, zip_path)
        if not verify_zip(zip_path):
            raise RuntimeError(
                "Downloaded zip file is still corrupted. "
                "Please download the PlantVillage dataset manually and extract it to: "
                f"{target_dir}"
            )
    
    extract_temp = zip_path.parent / "temp_extracted"
    if extract_temp.exists():
        shutil.rmtree(extract_temp)
    
    print("Extracting dataset zip file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_temp)
    
    # Locate color directory inside spMohanty repo layout (raw/color or color)
    found_color_dir = None
    for root, dirs, files in os.walk(extract_temp):
        if Path(root).name == "color" and len(dirs) > 0:
            found_color_dir = Path(root)
            break
    
    if not found_color_dir:
        # Fallback: search for any directory with class subdirectories
        for root, dirs, files in os.walk(extract_temp):
            if len(dirs) > 5:
                found_color_dir = Path(root)
                break

    if not found_color_dir or not found_color_dir.exists():
        # Clean up temp dir before raising
        if extract_temp.exists():
            shutil.rmtree(extract_temp)
        raise RuntimeError("Could not find image folder in extracted archive.")

    print(f"Found image classes at: {found_color_dir}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    shutil.copytree(found_color_dir, target_dir)
    print(f"Dataset successfully organized at: {target_dir}")
    
    # Clean up temp files
    if extract_temp.exists():
        shutil.rmtree(extract_temp)
    if zip_path.exists():
        os.remove(zip_path)

def verify_dataset(target_dir: Path):
    """Print dataset class breakdown and image count statistics."""
    if not target_dir.exists():
        print(f"Dataset directory {target_dir} does not exist.")
        return False
    
    class_dirs = [d for d in target_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print("No class directories found.")
        return False
    
    print("\n" + "="*50)
    print(f" PLANT VILLAGE DATASET VERIFICATION ")
    print("="*50)
    print(f"Total Disease/Healthy Classes: {len(class_dirs)}")
    
    total_images = 0
    class_counts = {}
    for c_dir in sorted(class_dirs):
        images = [f for f in c_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        count = len(images)
        class_counts[c_dir.name] = count
        total_images += count
        print(f"  - {c_dir.name}: {count} images")
        
    print("-" * 50)
    print(f"Total Dataset Images: {total_images}")
    print("=" * 50 + "\n")
    return True

def get_dataset():
    """Main pipeline for ensuring dataset is ready."""
    if TARGET_DIR.exists() and verify_dataset(TARGET_DIR):
        print("Dataset already exists and verified!")
        return TARGET_DIR

    if not ZIP_PATH.exists():
        download_file(DATASET_URL, ZIP_PATH)
    
    extract_and_organize(ZIP_PATH, TARGET_DIR)
    verify_dataset(TARGET_DIR)
    return TARGET_DIR

if __name__ == "__main__":
    get_dataset()
