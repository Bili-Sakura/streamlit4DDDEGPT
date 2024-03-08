# Use the official Python 3.8 image as the base image
FROM python:3.8

WORKDIR /dde

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Bili-Sakura/streamlit4DDDEGPT.git .

# Install Conda environment dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]