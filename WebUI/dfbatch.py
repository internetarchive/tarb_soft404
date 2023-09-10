import io
import concurrent.futures
import time
import os , sys
import gc
import re 
import requests
import csv
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.exceptions import ConnectionError
from urllib3.exceptions import NewConnectionError
import nltk
from nltk.corpus import wordnet
from goose3 import Goose

g=Goose()
i=0

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


def main(input_file, output_file, error_file, index):
    
    INPUTFILE = input_file
    OUTPUTFILE = output_file
    OUTPUTFILE_ERROR = error_file
    ERROR_404 = error_file_website
    INDEX = int(index)
    
    with open(OUTPUTFILE, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Url", "size_total", "Load_time", "image_count", "image_unreachable",
                     "Average_word_length", "Ratio", "image_size_total", "matched_keywords",
                     "number_of_words", "Script_size", "Css_size", "response_status_code","Title","Article","Text","Image_Name"])

    with open(OUTPUTFILE_ERROR, "w", newline='') as fer:
        writer = csv.writer(fer, delimiter=',')
        writer.writerow(["Url","Text"])
    
    with open(ERROR_404, "w", newline='') as ferror:
        writer = csv.writer(ferror, delimiter=',')
        writer.writerow(["Url","HTTPS_status"])



    def write_details_to_csv_error(details):
        with open(ERROR_404, "a", newline='') as ferror:
            writer = csv.writer(ferror, delimiter=',')
            print("aa")
            writer.writerow([
                f'"{details["Url"]}"' ,f'"{details["Text"]}"'          
                ])
    def write_details_to_csv_error404(details):
        with open(OUTPUTFILE_ERROR, "a", newline='') as fer:
            writer = csv.writer(fer, delimiter=',')
            writer.writerow([
                f'"{details["Url"]}"' ,f'"{details["response_status_code"]}"'          
                ])

    def write_details_to_csv(details):
        with open(OUTPUTFILE, "a", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                f'"{details["Url"]}"', details["size_total"], details["Load_time"], details["image_count"],
                details["image_unreachable"], details["Average_word_length"],
                details["Ratio"], details["image_size_total"], details["matched_keywords"],
                details["number_of_words"], details["Script_size"], details["Css_size"],
                details["response_status_code"],f'"{"".join(details["Title"])}"',
                f'"{"".join(details["article"])}"',
                f'"{"".join(details["Text"])}"',
                f'"{"".join(details["Image_Name"])}"'
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
            if(response.status_code>200):
                details={
                    "Url":url,
                    "response_status_code":response.status_code
                }
                write_details_to_csv_error404(details)
                return
            
            size_total = len(response.content)

            # Get the number of images on the website.
            content = response.content.decode('utf-8', 'ignore')
            soup = BeautifulSoup(content, "html.parser")
            images = soup.find_all("img")
            image_count = len(images)


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
            words = soup.find_all(string=True)
            word_count = len(words)
            word_length_sum = sum([len(word) for word in words])
            average_word_length = word_length_sum / word_count


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
            title = soup.find("title")
            title_text = sanitize_csv_text(title.text)

            words = title_text.split()
            number_of_words = len(words)
            prob=0

            words = soup.find_all(string=True)
            word_count = len(words)
            word_length_sum = sum([len(word) for word in words])
            image_unreachable = 0
            image_size_total = 0
            image_tags = soup.find_all("img")
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
            content_bytes = size_total - image_size_total
            ratio = content_bytes / size_total

            #print(url, size_total, response.status_code, response1) uncomment to debug

            details = {
                "Url": url,
                "size_total": size_total,
                "Load_time": end_time - start_time,
                "image_count": image_count,
                "image_unreachable": image_unreachable,
                "Average_word_length":average_word_length,
                "Ratio": ratio,
                "image_size_total": image_size_total,
                "matched_keywords": matched_keywords,
                "number_of_words": number_of_words,
                "Script_size": script_sizes,
                "Css_size": css_sizes,
                "response_status_code": response.status_code,
                "prob":prob,
                "Title":title_text,
                "article":context,
                "Text":context_t,
                "Image_Name":meaningful_image_names

            }

            write_details_to_csv(details)
        except Exception as e:
            print(f"Error processing website {url}: {str(e)}")

            details = {
                "Url": url,
                "Text":f"{str(e)}"
            }
            write_details_to_csv_error(details)


    # Create a list of websites.
    df = pd.read_csv(INPUTFILE)
    weburl = df.iloc[:, INDEX]
    websites = weburl.tolist()
    j = 0


    with open(OUTPUTFILE, "a", newline='') as f:
       
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_website_details, website) for website in websites]

            for future in concurrent.futures.as_completed(futures):
                j += 1
                if(j%50==0):
                    gc.collect()
                if (j%1==0):
                    print(f"Website {j} processed.")

    print("All websites processed.")
    print("missed",i)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python your_script.py <input_file> <output_file> <error_file> <index>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        error_file = sys.argv[3]
        error_file_website = sys.argv[4]
        index = sys.argv[5]
        main(input_file, output_file, error_file, index)
