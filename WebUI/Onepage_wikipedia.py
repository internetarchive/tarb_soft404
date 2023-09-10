import requests
import sys
import subprocess
import concurrent.futures
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from get_links import get_links


def display_iframe(url):
    iframe_html = f'<iframe src="{url}" width="400" height="300"></iframe>'
    st.write(iframe_html, unsafe_allow_html=True)


def run_catboost(url):
    
    cmd_catboost = ["python", "catboost_prediction.py",  '"' + url + '"']
    process_catboost = subprocess.run(cmd_catboost, capture_output=True, text=True)
    return process_catboost.stdout.strip(), process_catboost.stderr.strip()

def run_bert(url):
    cmd_bert = ["python", "bert.py", '"' + url + '"']
    process_bert = subprocess.run(cmd_bert, capture_output=True, text=True)
    return process_bert.stdout.strip(), process_bert.stderr.strip()

def run_app():
    st.title("Website Analysis App")

    # Create an input field for the website URL
    url = st.text_input("Enter the website URL")

    # Create a button to trigger processing
    if st.button("Analyze"):
        links = get_links(url)
        print(links)

        for link in links:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit the tasks to the executor
                future_catboost = executor.submit(run_catboost, link)
                future_bert = executor.submit(run_bert, link)
    
                # Retrieve the results when they are done
                output1, error1 = future_catboost.result()
                output2, error2 = future_bert.result()
    
            # Display the outputs or error messages
            st.write(link)
            st.write("CatBoost Output:", output1)
            
            st.write("BERT Output:", output2)

    
            display_iframe(link)

if __name__ == "__main__":
    run_app()
