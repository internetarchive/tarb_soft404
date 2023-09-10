import requests
from bs4 import BeautifulSoup
import csv
from PIL import Image
from urllib.parse import urljoin
import re
import io
import pandas as pd
from requests.exceptions import ConnectionError
from urllib3.exceptions import NewConnectionError
from soft404_duplicate import is_dead
import nltk
from nltk.corpus import wordnet
import io
import concurrent.futures
import time
import os , sys
#from soft404 import Soft404Classifier
import gc
import re 
from goose3 import Goose

#nltk.download('wordnet')

INPUTFILE=sys.argv[1]
OUTPUTFILE=sys.argv[2]
OUTPUTFILE_ERROR=sys.argv[3]
INDEX=int(sys.argv[4])
MAX=int(sys.argv[5])

g=Goose()

#clf = Soft404Classifier()
i=0
with open(OUTPUTFILE, "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Url", "size_total", "Load_time", "image_count", "image_unreachable",
                     "average_word_length", "ratio", "image_size_total", "matched_keywords",
                     "number_of_words", "Script_size", "Css_size", "response_status_code","Title","Article","Text","Image_Name"])

with open(OUTPUTFILE_ERROR, "w", newline='') as fer:
    writer = csv.writer(fer, delimiter=',')
    writer.writerow(["Url","Text"])



def write_details_to_csv_error(details):
    # Open the CSV file in append mode
    with open(OUTPUTFILE_ERROR, "a", newline='') as fer:
        writer = csv.writer(fer, delimiter=',')
        writer.writerow([
            f'"{details["Url"]}"' ,f'"{details["Text"]}"'          
            ])

def write_details_to_csv(details):
    # Open the CSV file in append mode
    with open(OUTPUTFILE, "a", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([
            f'"{details["Url"]}"', details["size_total"], details["Load_time"], details["image_count"],
            details["image_unreachable"], details["average_word_length"],
            details["ratio"], details["image_size_total"], details["matched_keywords"],
            details["number_of_words"], details["Script_size"], details["Css_size"],
            details["response_status_code"],details["prob"],details["is_dead"],f'"{"".join(details["Title"])}"',
            f'"{"".join(details["article"])}"',
            f'"{"".join(details["Text"])}"',
            f'"{"".join(details["Image_Name"])}"'
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




def get_file_size(url,timeout=30):
    try:
        response = requests.head(url,timeout=30)
        if response.status_code == 200:
            return int(response.headers.get('content-length', 0))
        return 0
    except requests.exceptions.RequestException as e:
        i=i+1
        #print("Error retrieving file size:", e)
        return 0


def get_website_details(url):
    url = url.strip('"')
    start_time = time.time()
    try:
        response = requests.get(url, timeout=30)
        end_time = time.time()
        size_total = len(response.content)

        # Get the number of images on the website.
        content = response.content.decode('utf-8', 'ignore')
        soup = BeautifulSoup(content, "html.parser")
        images = soup.find_all("img")
        image_count = len(images)
        res=is_dead(url)


        #para
        g = Goose()
        article = g.extract(raw_html=content)  
        context=sanitize_csv_text(article.cleaned_text[:200])

        paragraphs = soup.find_all("p")
        extracted_text = []
        for paragraph in paragraphs:
            text = paragraph.get_text()
            extracted_text.append(text)
        context_t = " ".join(extracted_text)
        #print(context_t)

        context_t=sanitize_csv_text(context_t)

        
        ##image to be optimised
        images = soup.find_all("img")
        k=0
        image_names = []
        for i, image in enumerate(images):
           if i < 5:  # Limit to top 5 images
               image_url = image["src"]
               image_name = os.path.basename(image_url)
               if len(image_name) > 25:
                   image_name = image_name[:25]  # Truncate name to 25 characters
               image_names.append(image_name)
           else:
               break
        meaningful_image_names = []
        for i, image in enumerate(images):
            if i < 5:  # Limit to top 5 images
                image_url = image["src"]
                image_name = os.path.basename(image_url)

                if is_meaningful(image_name):
                    if len(image_name) > 25:
                        image_name = image_name[:25]  # Truncate name to 25 characters
                    meaningful_image_names.append(image_name)
            else:
                break
        #print(meaningful_image_names)    
        #meaningful_image_names=sanitize_csv_text(meaningful_image_names)
        # Get the number of words in the title.
        title = soup.find("title")
        title_text = sanitize_csv_text(title.text)

        words = title_text.split()
        number_of_words = len(words)
        prob=0#clf.predict(title_text)
        # Get the average word length on the website.
        words = soup.find_all(string=True)
        word_count = len(words)
        word_length_sum = sum([len(word) for word in words])
        #average_word_length = word_length_sum / word_count
        
        image_unreachable = 0

        image_size_total = 0

        # Find all image elements
        image_tags = soup.find_all("img")

        # Get the base URL
        base_url = response.url

        # Iterate over the image elements and get their sizes
        for img_tag in image_tags:
            if "src" in img_tag.attrs:
                img_url = img_tag["src"]
                absolute_img_url = urljoin(base_url, img_url)
                try:
                    response = requests.get(absolute_img_url, stream=True)
                    image_data = io.BytesIO(response.content)
                    image = Image.open(image_data)
                    # Get the size of the image in bytes
                    image_size = image_data.getbuffer().nbytes
                    image_size_total += image_size
                except Exception as e:
                    image_unreachable += 1

        script_tags = soup.find_all('script')

        # Retrieve the sizes of script files
        script_sizes = 0
        for script_tag in script_tags:
            src = script_tag.get('src')
            if src:
                script_url = url + src if not src.startswith('http') else src
                script_size = get_file_size(script_url)
                script_sizes += script_size

        css_links = soup.find_all('link', rel='stylesheet')

        # Retrieve the sizes of CSS files
        css_sizes = 0
        for css_link in css_links:
            href = css_link.get('href')
            if href:
                css_url = url + href if not href.startswith('http') else href
                css_size = get_file_size(css_url)
                css_sizes += css_size

        keywords = ['404', 'error', 'found', 'page', 'file', 'resource', 'url', 'access denied', 'forbidden',
                    'unavailable', 'gone', 'redirect', 'temporary', 'permanent', 'server', 'client', 'browser',
                    'network', 'dns', 'typo', 'misspelling', 'stale', 'broken', 'dead', 'obsolete', 'expired',
                    'invalid', 'malformed', 'corrupt', 'unreachable', 'deleted']

        pattern = re.compile(fr'\b({"|".join(re.escape(kw) for kw in keywords)})\b', re.IGNORECASE)

        matched_elements = soup.find_all(string=pattern)
        matched_keywords = len(matched_elements)

        # Get the ratio of content bytes to total bytes.
        content_bytes = size_total - image_size_total
        ratio = content_bytes / size_total

        #print(url, size_total, response.status_code, response1)

        details = {
            "Url": url,
            "size_total": size_total,
            "Load_time": end_time - start_time,
            "image_count": image_count,
            "image_unreachable": image_unreachable,
            "average_word_length":0,# average_word_length,
            "ratio": ratio,
            "image_size_total": image_size_total,
            "matched_keywords": matched_keywords,
            "number_of_words": number_of_words,
            "Script_size": script_sizes,
            "Css_size": css_sizes,
            "response_status_code": response.status_code,
            "prob":prob,
            "is_dead":is_dead,
            "Title":title_text,
            "article":context,
            "Text":context_t,
            "Image_Name":meaningful_image_names

        }
        #tt=time.time()
        #print(details)
        write_details_to_csv(details)
        #tt1=time.time()
        #print("The time teken is ")
        #print(tt1-start_time)
        #print(tt-start_time)

    except Exception as e:
        #i=i+1
        print(f"Error processing website {url}: {str(e)}")

        details = {
            "Url": url,
            "Text":f"Error processing website {url}: {str(e)}"
        }
        write_details_to_csv_error(details)


# Create a list of websites.
df = pd.read_csv(INPUTFILE)
weburl = df.iloc[:, INDEX]
websites = weburl.tolist()
j = 0

# Create a CSV file to store the data.


with open(OUTPUTFILE, "a", newline='') as f:
    #writer = csv.writer(f)
    #writer.writerow(["Url", "size_total", "Load_time", "image_count", "image_unreachable",
    #                 "Average_word_length", "Ratio", "image_size_total", "matched_keywords",
    #                 "number_of_words", "Script_size", "Css_size", "response_status_code","Title","Text","Image_Name"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX) as executor:
        futures = [executor.submit(get_website_details, website) for website in websites]

        for future in concurrent.futures.as_completed(futures):
            j += 1
            if(j%50==0):
                gc.collect()
            if (j%1==0):
                print(f"Website {j} processed.")

        # Add a final step to make sure the details of the last website are written
        #for future in concurrent.futures.as_completed(futures):
        #    try:
        #        future.result()
        ##        print(future.result())
        #    except Exception as e:
        #        print(f"Error processing website: {str(e)}")

print("All websites processed.")
print("missed",i)
