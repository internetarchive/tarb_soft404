FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the contents of the local project directory to the container
COPY WebUI ./

CMD ["streamlit", "run", "WebUI.py"]
