import os
import sys
import requests
import subprocess
import pandas as pd
import streamlit as st
import concurrent.futures
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from get_links import get_links


def run_catboost(url):
    cmd_catboost = ["python3", "Catboost_prediction.py",  '"' + url + '"']
    process_catboost = subprocess.run(cmd_catboost, capture_output=True, text=True)
    return process_catboost.stdout.strip(), process_catboost.stderr.strip()

def run_bert(url):
    cmd_bert = ["python3", "bert.py", '"' + url + '"']
    process_bert = subprocess.run(cmd_bert, capture_output=True, text=True)
    return process_bert.stdout.strip(), process_bert.stderr.strip()



st.title("Soft-404 Detection")
selection = st.sidebar.radio("Choose the setting ", ("Wikipedia Page Analyser", "Single Page Analyse", "Bulk Analyser"))

if selection == "Wikipedia Page Analyser":
    st.subheader("Wikipedia Page Analyser")
  
   
    def display_iframe(url):
        iframe_html = f'<iframe src="{url}" width="400" height="300"></iframe>'
        st.write(iframe_html, unsafe_allow_html=True)

    url = st.text_input("Enter the website URL")
    if st.button("Analyze"):
        links = get_links(url)
        print(links)
        for link in links:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_catboost = executor.submit(run_catboost, link)
                future_bert = executor.submit(run_bert, link)
                output1, error1 = future_catboost.result()
                output2, error2 = future_bert.result()
            st.write(link)
            st.write("CatBoost Output:", output1)
            st.write("BERT Output:", output2)
            display_iframe(link)
    

elif selection == "Single Page Analyse":
    st.subheader("Single Page Analyse")
    def display_iframe(url):
        iframe_html = f'<iframe src="{url}" width="800" height="600"></iframe>'
        st.write(iframe_html, unsafe_allow_html=True)
    url = st.text_input("Enter the website URL")
    if st.button("Analyze"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_catboost = executor.submit(run_catboost, url)
            future_bert = executor.submit(run_bert, url)
            output1, error1 = future_catboost.result()
            output2, error2 = future_bert.result()
        st.write("CatBoost Output:", output1)
        st.write("BERT Output:", output2)
        display_iframe(url)

elif selection == "Bulk Analyser":
    st.subheader("Bulk Analyser")
    def ext(df):
        temp_csv = "temp_df.csv"
        df.to_csv(temp_csv, index=False)
        cmd_ext = ["python", "dfbatch.py",temp_csv,"output1.csv","error_404.csv","error.csv","0" ]
        process_ext = subprocess.run(cmd_ext, capture_output=True, text=True)
        os.remove(temp_csv)
        return process_ext.stdout.strip(), process_ext.stderr.strip()

    def run_catboost():
        cmd_catboost = ["python", "catboost_df.py"]
        process_catboost = subprocess.run(cmd_catboost, capture_output=True, text=True)
        return 
    def run_bert():
        cmd_bert = ["python", "bert_df.py"]
        process_bert = subprocess.run(cmd_bert, capture_output=True, text=True)
        return 

    def display_iframe(url):
        st.write(url)
        iframe_html = f'<iframe src="{url}" width="400" height="300"></iframe>'
        st.write(iframe_html, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Original DataFrame")
        st.dataframe(df)
        if st.button("Process"):
            ext(df)
            st.success("Processing complete!")
        if os.path.exists("output1.csv") and os.path.exists("error.csv") and os.path.exists("error_404.csv"):
            st.subheader("Data Extracted")
            df1 = pd.read_csv("output1.csv", encoding="cp1252")
            st.dataframe(df1)

            st.subheader("Errors_webpages")
            df3 = pd.read_csv("error_404.csv", encoding="cp1252")
            st.dataframe(df3)

            st.subheader("Errors")
            df2 = pd.read_csv("error.csv", encoding="cp1252")
            st.dataframe(df2)
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_catboost = executor.submit(run_catboost)
            future_bert = executor.submit(run_bert)
        if os.path.exists("output1_cat.csv") and os.path.exists("output1_bert.csv"):
            st.subheader("Predictions ")
            df1 = pd.read_csv("output1_cat.csv", encoding="cp1252")
            df2 = pd.read_csv("output1_bert.csv", encoding="cp1252")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Catboost Tree based model")
                st.dataframe(df1)

            with col2:
                st.subheader("Bert")
                st.dataframe(df2)

        urls = df["URL"].tolist()
        iframe_columns = []

