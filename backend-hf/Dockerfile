FROM nvidia/cuda:11.7.1-base-ubuntu20.04


# Layers are approximately ordered from lowest turnover to highest
# 1. Install Python & other base software

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential curl git
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    sh /tmp/miniconda.sh -b -p /opt/anaconda3


# 2. Install Python dependencies

RUN /opt/anaconda3/bin/conda install -y pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

WORKDIR /opt/sandle
COPY requirements.txt /opt/sandle/
RUN /opt/anaconda3/bin/conda install -y pip && \
    /opt/anaconda3/bin/conda run --no-capture-output pip install -r /opt/sandle/requirements.txt


# 3. Add large model files


# 4. Add application code

COPY serve-backend-hf.py /opt/sandle/


# 5. Perform remaining configuration

ENV LANG="C.UTF-8"
EXPOSE 8000
ENTRYPOINT ["/opt/anaconda3/bin/conda", "run", "--no-capture-output", "python", "serve-backend-hf.py"]


# Sentry configuration (needed by Python at runtime)

ARG SENTRY_DSN
ARG SENTRY_RELEASE

ENV SENTRY_DSN=${SENTRY_DSN}
ENV SENTRY_RELEASE=${SENTRY_RELEASE}
