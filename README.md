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
