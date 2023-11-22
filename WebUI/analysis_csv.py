import streamlit as st
import subprocess
import pandas as pd
import os
import concurrent.futures


def ext(df):
    temp_csv = "temp_df.csv"
    df.to_csv(temp_csv, index=False)
    cmd_ext = ["python3", "dfbatch.py",temp_csv,"output1.csv","error_404.csv","error.csv","0" ]
    process_ext = subprocess.run(cmd_ext, capture_output=True, text=True)
    os.remove(temp_csv)
    return process_ext.stdout.strip(), process_ext.stderr.strip()

def run_catboost():
    cmd_catboost = ["python3", "catboost_df.py"]
    process_catboost = subprocess.run(cmd_catboost, capture_output=True, text=True)
    return 
def run_bert():
    cmd_bert = ["python3", "bert_df.py"]
    process_bert = subprocess.run(cmd_bert, capture_output=True, text=True)
    return 

def display_iframe(url):
    st.write(url)
    iframe_html = f'<iframe src="{url}" width="400" height="300"></iframe>'
    st.write(iframe_html, unsafe_allow_html=True)




def main():
    st.title("Soft404 Analysis")

    # Upload DataFrame CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display original DataFrame
        st.subheader("Original DataFrame")
        st.dataframe(df)

        # Process DataFrame when the user clicks the "Process" button
        if st.button("Process"):
            ext(df)
            st.success("Processing complete!")

        # Display processed CSV files if they exist
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
            # Submit the tasks to the executor
            future_catboost = executor.submit(run_catboost)
            future_bert = executor.submit(run_bert)
        if os.path.exists("output1_cat.csv") and os.path.exists("output1_bert.csv"):
            st.subheader("Predictions ")
            df1 = pd.read_csv("output1_cat.csv", encoding="cp1252")
            df2 = pd.read_csv("output1_bert.csv", encoding="cp1252")

            # Create two columns for displaying DataFrames side by side
            col1, col2 = st.columns(2)

            # Display df1 in the first column
            with col1:
                st.subheader("Catboost Tree based model")
                st.dataframe(df1)

            # Display df2 in the second column
            with col2:
                st.subheader("Bert")
                st.dataframe(df2)

        urls = df["URL"].tolist()
        iframe_columns = []
        ##Uncomment if u want to view websites in Iframes
        # Divide the urls into multiple columns
        # num_columns = 4  # Set the number of columns here
        # urls_per_column = (len(urls) + num_columns - 1) // num_columns

        # for i in range(0, len(urls), urls_per_column):
        #     column_urls = urls[i:i + urls_per_column]
        #     iframe_column = [f'<iframe src="{url}" width="400" height="300"></iframe>' for url in column_urls]
        #     iframe_columns.append(iframe_column)

# # Display iframes in multiple columns
#         with st.container():
#             for column in iframe_columns:
#                 col1, col2, col3, col4 = st.columns(num_columns)
#                 for iframe_html, col in zip(column, [col1, col2, col3, col4]):
#                     col.write(iframe_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
