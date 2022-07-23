FROM continuumio/miniconda3:4.10.3


# Layers are approximately ordered from lowest turnover to highest
# 1. Install Python & other base software

RUN apt-get update && \
    apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*


# 2. Install Python dependencies

WORKDIR /opt/openaisle
COPY requirements.txt /opt/openaisle/
RUN pip install -r requirements.txt


# 3. Add large model files


# 4. Add application code

COPY serve-openai-wrapper.py /opt/openaisle/


# 5. Perform remaining configuration

EXPOSE 8000
ENTRYPOINT ["python", "serve-openai-wrapper.py"]
