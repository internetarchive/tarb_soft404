import subprocess
import os

# make a dir for models

def make_dir(path):
    try:
        os.makedirs(path)
        print(f"Models Directory created: {path}")
    except OSError as e:
        print(f"Error making model dir at path {path} due to {e}")


def download_models(url,output_file):
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return
    try:
        subprocess.run(['wget', url, '-O', output_file], check=True)
        print(f"Downloaded :{output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")

def install_lib():
    try:
        subprocess.run(['pip3','install','-r','requirements-cpu.txt'])
        print(f"Libraries installed")
    except subprocess.CalledProcessError as e:
        print (f"Error {e}")


try:
    import torch
    subprocess.run(['pip3','uninstall','torch'])
except:
    print(f"No previous versions of pytorch found")

install_lib()

path ="Models"
make_dir(path)

url_bert= "https://archive.org/download/tarb-gsoc-2023-soft-404/TARB_GSoC23_Soft404analysis/Models/bert_model1"
outputfile_bert = "Models/bert_model1"

url_catboost= "https://archive.org/download/tarb-gsoc-2023-soft-404/TARB_GSoC23_Soft404analysis/Models/catboost_model.bin"
outputfile_catboost="Models/catboost_model.bin"

download_models(url_bert,outputfile_bert)
download_models(url_catboost,outputfile_catboost)

import nltk
nltk.download('wordnet')
