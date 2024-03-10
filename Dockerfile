# streamlit4DDEGPT/Dockerfile

# Use the official Python 3.8 image as the base image
FROM python:3.8

# Set the working directory in the container to /myapp
WORKDIR /myapp

# Install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# # Remove existing files in the container at /myapp (if needed)
RUN rm -rf /myapp/*

ADD . /myapp

RUN pip install --no-cache-dir -r requirements.txt


# Run streamlit when the container launches
CMD ["streamlit", "run", "app.py"]