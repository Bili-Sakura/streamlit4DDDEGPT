# streamlit4DDEGPT/Dockerfile

# Only used in first Initlization

# Use the official Python 3.8 image as the base image
FROM python:3.8

# Set the working directory in the container to /myapp
WORKDIR /myapp

# # Remove existing files in the container at /myapp (if needed)
RUN rm -rf /myapp/*

# Install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b docker --depth 1 https://github.com/Bili-Sakura/streamlit4DDDEGPT.git .

RUN pip install --no-cache-dir -r requirements.txt

# Run streamlit when the container launches
CMD ["streamlit", "run", "app.py"]