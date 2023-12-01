# tarb_gsoc23_soft404
  

## Overview
This repository is a comprehensive toolset for soft 404 detection, encompassing data scraping, model training, web user interfaces, and inference capabilities. It utilizes tree-based models and Transformers like BERT to analyze webpage structure and content, making it a versatile solution for identifying soft 404 errors on websites.

## Usage

Clone the repository

```sh
git clone https://github.com/internetarchive/tarb_soft404.git
```

Run the setup file  based on machine configuration i.e. Cpu or Gpu
```sh
python3 setup-gpu.py 
```

To create Docker Image  and run it

```sh
docker build -t soft-404 .
docker run soft-404
```
Running Web UI
```sh
streamlit run WebUI.py
```

## Directory Structure

  

-  `Datasets/`: This directory houses Python scripts responsible for data extraction from websites, resulting in the creation of datasets in CSV format. It also includes helper tools for breaking down files into smaller CSV files and ensuring that only one website per domain is included in the datasets. There are two extraction methods available: one based on the structure of webpages and another based on their content.

-  `Train/`: In this directory, you'll find files for training two types of models:

	-   A tree-based Catboost model that focuses on analyzing the structure of webpages.
	-   A BERT model that has been fine-tuned for content-based analysis of webpages.
 	-   Trained models are available at [models](https://archive.org/download/tarb-gsoc-2023-soft-404/TARB_GSoC23_Soft404analysis/Models/)
 	   		
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
- pip install -r requirements.txt



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

6. **dfbatch.py**: This script takes in csv and return both structural and content based analysis.
 `python run.py <inputfile.csv> <outputfilename.csv> <index of URL> <max_workers> `

7. **soft404_duplicate.py**: This is a classifier based upon if content of websites returned is in original URL and last part of URL changed to some random 25 letters.

### Train
1. **BERT_model**: This is a Jupyter Notebook file to preprocess the extracted data, train and save the trained BERT model, further we can also test the model if we have more labeled extracted data.

2. **Catboost_train**: This is a Jupyter Notebook file to preprocess the extracted data, train a catboost model and save it. Further we can also test the model if we have more labeled extracted data.

3. **run.py**: It is same file as BERT_model but in form of a python script to train on non GUI machine, GPU servers.
   `python run.py`

4. **test.py**: This file load the fine-tuned BERT model and makes prediction and according to the label and prediction generates the score like accuracy precision and recall along with confusion matrix.
   `python test.py`

### WebUI
1. **Single_page_analysis.py**: This is a webUI script built on streamlit , given one link this will return predictions whether its 200 okay or if 404 error.
 `streamlit run Single_page_analysis.py`
2. **Onepage_wikipedia.py**: This is a webUI script built on streamlit , given one Wikipedia link this will return predictions whether its 200 okay or if 404 error on all non-wiki and non-archive links present on the page.
  `streamlit run Onepage_wikipedia.py``
3. **analysis_csv.py**: This is a webUI script built on streamlit , given one csv files of links it will make prediction on all links present and which return HTTPS status code of 200 okayy else it stores non 200 okay URLS in one csv and all other websites which give errors while scraping in other.
  `streamlit run analysis_csv.py`

4. **bert.py**: This is a helper file given a link it returns BERT predictions
   `python bert.py '<URL>'`
5. **bert_df.py**: This is a helper file which takes in a csv dataframe of extracted data and returns BERT predictions csv.
6. **Catboost_prediction.py**:This is a helper file given a link it returns catboost predictions
  `python Catboost_prediction.py '<URL>'`
8. **catboost_df.py**:This is a helper file which takes in a csv dataframe of extracted data and returns catboost predictions csv.
9.  **dfbatch.py**: This script takes in csv and return both structural and content based analysis.
 `python run.py <inputfile.csv> <outputfilename.csv> <index of URL> <max_workers> `

10. **get_links.py**: This is used to extract all non wikipedia and no archive links from a webpage (wikipedia webpage)
 `from get_links import get_links` `get_links(URL)` 

### Inference
1. **bert.py**: This python script takes in a URL scrapes it in terms of content passes through BERT  and returns predictions `python bert.py '<URL>'`

2. **bert_df.py**: This is a python script file which takes in a csv dataframe of extracted data and returns BERT predictions csv.

3. **Catboost_prediction.py**: This python script takes in a URL scrapes it in terms of structure passes it through catboost  and return predictions `python Catboost_prediction.py '<URL>'`

4. **catboost_df.py**: This is a python script which takes in a csv dataframe of extracted data and returns catboost predictions csv.
