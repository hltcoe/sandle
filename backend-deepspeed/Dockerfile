FROM nvidia/cuda:11.4.1-devel-ubuntu20.04


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

RUN pip install torch==1.13 torchvision==0.14

RUN git clone -b v0.9.0 https://github.com/microsoft/DeepSpeed.git /opt/deepspeed-src && \
    pip install /opt/deepspeed-src && \
    pip install deepspeed-mii triton==1.0.0 && \
    rm -rf /opt/deepspeed-src

WORKDIR /opt/sandle
COPY requirements.txt /opt/sandle/
RUN pip install -r /opt/sandle/requirements.txt


# 3. Add large model files


# 4. Add application code

COPY backend_deepspeed /opt/sandle


# 5. Perform remaining configuration

ENV LANG="C.UTF-8"
EXPOSE 8000
ENTRYPOINT ["bash", "docker-command.bash"]


# Sentry configuration (needed by Python at runtime)

ARG SENTRY_DSN
ARG SENTRY_RELEASE

ENV SENTRY_DSN=${SENTRY_DSN}
ENV SENTRY_RELEASE=${SENTRY_RELEASE}
