import streamlit as st
import subprocess
import concurrent.futures


def display_iframe(url):
    iframe_html = f'<iframe src="{url}" width="800" height="600"></iframe>'
    st.write(iframe_html, unsafe_allow_html=True)




def run_catboost(url):
    cmd_catboost = ["python3", "Catboost_prediction.py", '"' + url + '"']
    process_catboost = subprocess.run(cmd_catboost, capture_output=True, text=True)
    return process_catboost.stdout.strip(), process_catboost.stderr.strip()

def run_bert(url):
    cmd_bert = ["python3", "bert.py",  '"' + url + '"']
    process_bert = subprocess.run(cmd_bert, capture_output=True, text=True)
    return process_bert.stdout.strip(), process_bert.stderr.strip()

def run_app():
    st.title("Website Analysis App")


    url = st.text_input("Enter the website URL")


    if st.button("Analyze"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the tasks to the executor
            future_catboost = executor.submit(run_catboost, url)
            future_bert = executor.submit(run_bert, url)


            output1, error1 = future_catboost.result()
            output2, error2 = future_bert.result()

        # Display the outputs or error messages
        st.write("CatBoost Output:", output1)
        st.write("BERT Output:", output2)
        display_iframe(url)

if __name__ == "__main__":
    run_app()
