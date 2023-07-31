FROM nvcr.io/nvidia/pytorch:23.04-py3
RUN pip install vllm
EXPOSE 8000
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
