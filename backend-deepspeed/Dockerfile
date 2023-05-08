FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04


# Layers are approximately ordered from lowest turnover to highest
# 1. Install Python & other base software

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        git \
        libaio-dev \
        python-is-python3 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*


# 2. Install Python dependencies

WORKDIR /opt/sandle
COPY requirements.txt /opt/sandle/
RUN pip install -r /opt/sandle/requirements.txt


# 3. Add large model files


# 4. Add application code

COPY \
    docker-command.bash \
    main.py \
    models.json \
    /opt/sandle/


# 5. Perform remaining configuration

ENV LANG="C.UTF-8"
EXPOSE 8000
ENTRYPOINT ["bash", "docker-command.bash"]


# Sentry configuration (needed by Python at runtime)

ARG SENTRY_DSN
ARG SENTRY_RELEASE

ENV SENTRY_DSN=${SENTRY_DSN}
ENV SENTRY_RELEASE=${SENTRY_RELEASE}
