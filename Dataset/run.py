import re
import io
import csv
import time
import os , sys
import gc
import requests
import pandas as pd

from bs4 import BeautifulSoup
from PIL import Image
import concurrent.futures
from urllib.parse import urljoin
from requests.exceptions import ConnectionError
from urllib3.exceptions import NewConnectionError

#from soft404 import Soft404Classifier
from soft404_duplicate import is_dead



INPUTFILE=sys.argv[1]
OUTPUTFILE=sys.argv[2]
INDEX=int(sys.argv[3])

# clf = Soft404Classifier()
i=0
def write_details_to_csv(details):
    # Open the CSV file in append mode
    with open(OUTPUTFILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            details["Url"], details["size_total"], details["Load_time"], details["image_count"],
            details["image_unreachable"], details["average_word_length"],
            details["ratio"], details["image_size_total"], details["matched_keywords"],
            details["number_of_words"], details["Script_size"], details["Css_size"],
            details["response_status_code"], details["is_dead"],details["prob"]
        ])


def get_file_size(url,timeout=30):
    try:
        response = requests.head(url,timeout=30)
        if response.status_code == 200:
            return int(response.headers.get('content-length', 0))
        return 0
    except requests.exceptions.RequestException as e:
        i=i+1
        return 0


def get_website_details(url):
    start_time = time.time()
    try:
        response = requests.get(url, timeout=30)
        end_time = time.time()
        response1 = is_dead(url)
        size_total = len(response.content)

        # Get the number of images on the website.
        soup = BeautifulSoup(response.content, "html.parser")
        images = soup.find_all("img")
        image_count = len(images)

        # Get the number of words in the title.
        title = soup.find("title")
        title_text = title.text
        words = title_text.split()
        number_of_words = len(words)
        # prob=clf.predict(title_text)
        prob=0
        # Get the average word length on the website.
        words = soup.find_all(string=True)
        word_count = len(words)
        word_length_sum = sum([len(word) for word in words])
        average_word_length = word_length_sum / word_count

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

        print(url, size_total, response.status_code, response1)

        details = {
            "Url": url,
            "size_total": size_total,
            "Load_time": end_time - start_time,
            "image_count": image_count,
            "image_unreachable": image_unreachable,
            "average_word_length": average_word_length,
            "ratio": ratio,
            "image_size_total": image_size_total,
            "matched_keywords": matched_keywords,
            "number_of_words": number_of_words,
            "Script_size": script_sizes,
            "Css_size": css_sizes,
            "response_status_code": response.status_code,
            "is_dead": response1,
            "prob":prob
        }
      
        write_details_to_csv(details)
        

    except Exception as e:
        i=i+1
        print(f"Error processing website {url}: {str(e)}")
        


# Create a list of websites.
df = pd.read_csv(INPUTFILE)
weburl = df.iloc[:, INDEX]
websites = weburl.tolist()
j = 0

# Create a CSV file to store the data.
with open(OUTPUTFILE, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Url", "size_total", "Load_time", "image_count", "image_unreachable",
                     "average_word_length", "ratio", "image_size_total", "matched_keywords",
                     "number_of_words", "Script_size", "Css_size", "response_status_code", "is_dead"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=192) as executor:
        futures = [executor.submit(get_website_details, website) for website in websites]

        for future in concurrent.futures.as_completed(futures):
            j += 1
            if(j%50==0):
                gc.collect()
            if (j%1==0):
                print(f"Website {j} processed.")
