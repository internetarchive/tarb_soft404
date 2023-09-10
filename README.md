# tarb_gsoc23_soft404
  

## Overview
This repository is a comprehensive toolset for soft 404 detection, encompassing data scraping, model training, web user interfaces, and inference capabilities. It utilizes tree-based models and Transformers like BERT to analyze webpage structure and content, making it a versatile solution for identifying soft 404 errors on websites.


## Directory Structure

  

-  `Datasets/`: This directory houses Python scripts responsible for data extraction from websites, resulting in the creation of datasets in CSV format. It also includes helper tools for breaking down files into smaller CSV files and ensuring that only one website per domain is included in the datasets. There are two extraction methods available: one based on the structure of webpages and another based on their content.

-  `Train/`: In this directory, you'll find files for training two types of models:

	-   A tree-based Catboost model that focuses on analyzing the structure of webpages.
	-   A BERT model that has been fine-tuned for content-based analysis of webpages. 
	- Additionally, there are files for experimenting with hyperparameters of the Catboost model to optimize its performance. You'll also find functions for testing the trained models.

-  `WebUI/`: This directory contains web user interfaces (WebUI) built using Streamlit. These interfaces allow users to input data and receive predictions regarding whether a webpage is a soft 404 error or not. There are three WebUI applications:

	-   One that takes a single website as input and provides a result indicating whether it's healthy or not.
	-   Another one that accepts a Wikipedia page as input and provides the status of all non-Wikipedia and archive pages within that webpage.
	-   The third interface takes a list of websites in CSV format as input and processes them collectively.
 
-  `Inference/`:In this directory, you'll find files designed for making predictions about websites. These files enable prediction generation using both models (Catboost and BERT) and both types of inputs (structural and content-based).
  

## Prerequisites

  

- Python 3.x
- pip
- further library requirements are present in requirement.txt file.


### Dataset
1. **chunks.py**: This script divides large csv into small csv based upon chunk_size.
2. **duplicates.py**: This script takes a csv keeps one website per domain and returns the a processed csv.
3. **Content_extractor_new.py**: This script takes in a list of websites in form a csv and returns the content based features in form of a csv with columns as URL, Title, Article, Text, meaningful image names and label. 
 `python Content_extractor_new.py <inputfile.csv> <outputfilename.csv> <index of URL> <max_workers> `
 max_workers is used to multithread the process but keeping it very high also stalled the process due to limitations of download speed RAM and computing power , its suggested to play around its value and find a sweet spot. For kaggle its around 256-320.
4. **Content_extractor_error.py**: This script is used to generate artificial 404 by adding random string in place of string after last '/' in websites that respond with False in is_dead function. This takes in a list of websites in form a csv and returns the content-based features in form of a csv with columns as URL, Title, Article, Text, meaningful image names and label. 
 `python Content_extractor_error.py <inputfile.csv> <outputfilename.csv> <index of URL> <max_workers> `
 5. **run.py**:This script takes in a list of websites in form a csv and returns the structure-based features in form of a csv with columns Url, size_total, Load_time, image_count, image_unreachable,average_word_length, ratio,  image_size_total, matched_keywords,number_of_words,Script_size, Css_size and labels. 
`python run.py <inputfile.csv> <outputfilename.csv> <index of URL>`
 6.**dfbatch.py**: This script takes in csv and return both structural and content based analysis.
 `python run.py <inputfile.csv> <outputfilename.csv> <index of URL> <max_workers> `
7.**soft404_duplicate.py**: This is a classifier based upon if content of websites returned is in original URL and last part of URL changed to some random 25 letters.

