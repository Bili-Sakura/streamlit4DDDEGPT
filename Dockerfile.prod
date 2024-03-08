# streamlit4DDEGPT/Dockerfile

# Only used in first Initlization

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

RUN git clone -b docker --depth 1 https://github.com/streamlit/streamlit-example.git .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8503

# Run streamlit when the container launches
# We run conmand mannually!
# CMD ["streamlit", "run", "app.py"]