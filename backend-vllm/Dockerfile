FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential build-essential curl git python-is-python3 python3 python3-dev python3-pip
#RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -o /tmp/miniconda.sh && \
#    sh /tmp/miniconda.sh -b -p /opt/anaconda3


# 2. Install Python dependencies

#RUN /opt/anaconda3/bin/conda install -y pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
RUN pip install vllm

#RUN /opt/anaconda3/bin/conda install -y pip && \
#    /opt/anaconda3/bin/conda run --no-capture-output pip install -r /opt/sandle/requirements.txt


ENV LANG="C.UTF-8"
EXPOSE 8000
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
