FROM continuumio/miniconda3:4.10.3


RUN apt-get update && \
    apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*


RUN pip install flask supervisor waitress


WORKDIR /app
COPY run.bash serve-status.py supervisord.conf /app/


EXPOSE 8000
ENTRYPOINT ["bash", "run.bash"]
