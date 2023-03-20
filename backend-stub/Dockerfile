FROM python:3.9
WORKDIR /opt/sandle
COPY \
    requirements.txt \
    serve-backend.py \
    models.json \
    /opt/sandle/
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["python", "serve-backend.py"]
