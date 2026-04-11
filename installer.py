"""Setup script for the soft404 package.
Usage:
    python installer.py --cpu
    python installer.py --gpu
    
This Script will install the required libraries and download the necessary models for the soft404 package.
"""

import argparse
import subprocess
import sys
import os
import time 
import logging
import urllib.request

logging.basicConfig(level=logging.INFO)

MODELS = {
    "bert_model1": "https://archive.org/download/tarb-gsoc-2023-soft-404/TARB_GSoC23_Soft404analysis/Models/bert_model1",
    "catboost_model.bin": "https://archive.org/download/tarb-gsoc-2023-soft-404/TARB_GSoC23_Soft404analysis/Models/catboost_model.bin"
}
## broken links right now 

def check_uv():
    try:
        subprocess.run(['uv', '--version'], check=True, capture_output=True)
    except Exception:
        print("Installing uv in current environment...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "uv"], check=True)
        
def install_libs(variant: str):
    print(f"Installing libraries for {variant} variant via uv...")

    if variant == "gpu":
        cmd = ["uv", "pip", "install", "-e", ".[gpu]"]
    else:
        cmd = ["uv", "pip", "install", "-e", ".[cpu]"]

    subprocess.run(cmd, check=True)
    
def download_models(retries=3, delay=2):
    """Downloads the models from the provided URLs and saves them to the Models directory."""
    os.makedirs("Models", exist_ok=True)

    for model_name, url in MODELS.items():
        output_file = os.path.join("Models", model_name)

        if os.path.exists(output_file):
            print(f"File already exists: {output_file}")
            continue
            ## checksum can be added here to verify the integrity of the file if it already exists

        for attempt in range(1, retries + 1):
            try:
                print(f"Downloading {model_name} (attempt {attempt})...")
                urllib.request.urlretrieve(url, output_file)
                print(f"Downloaded: {output_file}")
                break  
            
            except Exception as e:
                print(f"Attempt {attempt} failed for {url}: {e}")

                if attempt == retries:
                    print(f"Failed to download after {retries} attempts: {model_name}")
                else:
                    time.sleep(delay)
            
def download_nltk_data():
    "Downloads the necessary NLTK data for the package."
    import nltk
    nltk.download('wordnet', quiet=True)
    

def main():
    parser = argparse.ArgumentParser(description="Setup script for the soft404 package.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cpu', action='store_true', help='Install the CPU variant of the package.')
    group.add_argument('--gpu', action='store_true', help='Install the GPU variant of the package.')
    args = parser.parse_args()
    
    variant = "cpu" if args.cpu else "gpu"
    
    check_uv()
    install_libs(variant)
    # download_models()
    download_nltk_data()
    
    print(f"Setup completed successfully! for variant: {variant}")
    
if __name__ == "__main__":
        main()