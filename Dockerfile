# streamlit4DDEGPT/Dockerfile

# Only used in first Initlization

# Use the official Python 3.8 image as the base image
FROM python:3.8

# Set the working directory in the container to /myapp
WORKDIR /myapp

# # Remove existing files in the container at /myapp (if needed)
RUN rm -rf /myapp/*

ADD . /myapp

RUN pip install --no-cache-dir -r requirements.txt

# Run streamlit when the container launches
CMD ["streamlit", "run", "app.py"]