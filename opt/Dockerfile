FROM nvidia/cuda:11.0.3-base-ubuntu20.04


# Layers are approximately ordered from lowest turnover to highest
# 1. Install Python & other base software

ENV DEBIAN_FRONTEND=noninteractive
# Hack: Update cuda gpg key because old base image key has been rotated
#RUN apt-key del 7fa2af80 && \
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y build-essential curl
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    sh /tmp/miniconda.sh -b -p /opt/anaconda3


# 2. Install Python dependencies

WORKDIR /opt/openaisle
COPY requirements.txt /opt/openaisle/
RUN /opt/anaconda3/bin/conda install -y pip && \
    /opt/anaconda3/bin/conda install -y pytorch torchvision torchaudio -c pytorch && \
    /opt/anaconda3/bin/conda run --no-capture-output pip install -r /opt/openaisle/requirements.txt


# 3. Add large model files


# 4. Add application code

COPY serve-opt.py /opt/openaisle/


# 5. Perform remaining configuration

ENV LANG="C.UTF-8"
EXPOSE 8000
ENTRYPOINT ["/opt/anaconda3/bin/conda", "run", "--no-capture-output", "python", "serve-opt.py"]
