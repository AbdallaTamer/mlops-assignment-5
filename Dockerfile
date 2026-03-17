FROM python:3.10-slim

# Accept the RUN_ID passed from GitHub Actions
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

RUN echo "Simulating deployment..." && \
    echo "Downloading model artifacts for MLflow Run ID: ${RUN_ID}"

CMD ["echo", "Production model container is ready!"]
