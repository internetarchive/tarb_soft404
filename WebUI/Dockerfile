FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the local project directory to the container
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "WebUI.py"]
