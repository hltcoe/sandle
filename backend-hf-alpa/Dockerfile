FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04


# Layers are approximately ordered from lowest turnover to highest
# 1. Install Python & other base software

ENV DEBIAN_FRONTEND=noninteractive
# Hack: Update cuda gpg key because old base image key has been rotated
#RUN apt-key del 7fa2af80 && \
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        cmake \
        git \
        pybind11-dev \
        python-is-python3 \
        python3 \
        python3-pip


# 2. Install Python dependencies

WORKDIR /opt/sandle
RUN pip install cupy-cuda110 numpy==1.22

COPY install-alpa.bash /opt/sandle/
RUN bash install-alpa.bash

COPY requirements.txt /opt/sandle/
RUN pip install -r /opt/sandle/requirements.txt


# 3. Add large model files


# 4. Add application code

COPY serve-backend-hf.py /opt/sandle/


# 5. Perform remaining configuration

ENV LANG="C.UTF-8"
EXPOSE 6379
EXPOSE 8000
#ENTRYPOINT ["bash", "docker-command.bash"]
CMD ["ray", "start", "--head"]


# Sentry configuration (needed by Python at runtime)

ARG SENTRY_DSN
ARG SENTRY_RELEASE

ENV SENTRY_DSN=${SENTRY_DSN}
ENV SENTRY_RELEASE=${SENTRY_RELEASE}
