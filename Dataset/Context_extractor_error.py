import time
import sys
import os
import gc
import re
import io
import csv
import requests
from goose3 import Goose
from bs4 import BeautifulSoup
from PIL import Image
from urllib.parse import urljoin
import pandas as pd
from requests.exceptions import ConnectionError
from urllib3.exceptions import NewConnectionError
#from soft404 import Soft404Classifier
from soft404_duplicate import is_dead
from soft404_duplicate import added_url
import concurrent.futures
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

#clf = Soft404Classifier()

INPUTFILE=sys.argv[1]
OUTPUTFILE=sys.argv[2]
INDEX=int(sys.argv[3])
MAX=int(sys.argv[4])
g = Goose()

with open(OUTPUTFILE, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Url","Title","Text","Article","Image_Name", "response_status_code","is_dead","prob"])


def write_details_to_csv(details):
    # Open the CSV file in append mode
    with open(OUTPUTFILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f'{details["Url"]}',
            f'{"".join(details["Title"])}',
            f'{"".join(details["Text"])}',
            f'{"".join(details["Article"])}',
            f'{"".join(details["Image_Name"])}',
            details["response_status_code"],
            details["is_dead"],
            details["Probablity"]
        ])

def is_meaningful(name):
    # Check if the image name contains a meaningful word
    for word in name.split('_'):  # Split image name by underscores
        if wordnet.synsets(word):
            return True
    return False

def sanitize_csv_text(text):
    # Remove common CSV delimiters (comma, semicolon, tab, pipe, colon, and space)
    sanitized_text = re.sub(r'[,\t;|: ]', ' ', text)
    # Replace multiple spaces with a single space
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text)
    # Split the text into words and take the first 100 words
    words = sanitized_text.split()[:100]
    # Join the first 100 words back into a single string
    sanitized_text = ' '.join(words)
    return sanitized_text

def get_website_details(url_good):
    url=added_url(url_good)   
    try:
        response = requests.get(url, timeout=30)
        response1 = is_dead(url)
        
        content = response.content
        soup = BeautifulSoup(content, "html.parser")
        
        
        article = g.extract(raw_html=content)        
       
        context=sanitize_csv_text(article.cleaned_text[:200])
        paragraphs = soup.find_all("p")
        extracted_text = []
        for paragraph in paragraphs:
            text = paragraph.get_text()
            extracted_text.append(text)
        content_t = " ".join(extracted_text)
        content_t=sanitize_csv_text(content_t)


        title = soup.find("title")
        title_text = title.text
        title_text=sanitize_csv_text(title_text)
        #print(title_text)
        # prob=clf.predict(title_text)
        prob=0


        images = soup.find_all("img")
        image_names = []
        for i, image in enumerate(images):
           if i < 5:  
               image_url = image["src"]
               image_name = os.path.basename(image_url)
               if len(image_name) > 25:
                   image_name = image_name[:25]  # Truncate name to 25 characters
               image_names.append(image_name)
           else:
               break
        meaningful_image_names = []
        for i, image in enumerate(images):
            if i < 5:  
                image_url = image["src"]
                image_name = os.path.basename(image_url)

                if is_meaningful(image_name):
                    if len(image_name) > 25:
                        image_name = image_name[:25]  # Truncate name to 25 characters
                    meaningful_image_names.append(image_name)
            else:
                break
                
        #print(url, response.status_code, response1)
        details = {
            "Url": url,
            "Title":title_text,
            "Text":context,
            "Article":content_t,
            "Image_Name":meaningful_image_names,
            "response_status_code": response.status_code,
            "is_dead": response1,
            "Probablity":prob
        }
        write_details_to_csv(details)

    except Exception as e:
        print(f"Error processing website {url}: {str(e)}")
       

# Create a list of websites.
df = pd.read_csv(INPUTFILE)
weburl = df.iloc[:, INDEX]
websites = weburl.tolist()
j = 0

# Create a CSV file to store the data.
with open(OUTPUTFILE, "w", newline='') as f:
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX) as executor:
        futures = [executor.submit(get_website_details, website) for website in websites]

        for future in concurrent.futures.as_completed(futures):
            j += 1
            if(j%50==0):
                gc.collect()
            print(f"Website {j} processed.")
