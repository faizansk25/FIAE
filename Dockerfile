# FIAE container image — cloud-native / Kubernetes deployment
#
# Build:  docker build -t fiae:latest .
# Run:    docker run --rm -v ${PWD}:/data fiae:latest inspect /data/data.csv
# K8s:    see k8s/ deployment manifests (or any Job/CronJob running `fiae ...`)
#
# Includes tier1 deps (numpy, scikit-learn, polars) plus cloud SDKs so
# s3://, gs://, and az:// sources work out of the box.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE MANIFEST.in setup.py ./
COPY src ./src

RUN pip install --no-cache-dir ".[tier1]" \
    && pip install --no-cache-dir boto3 google-cloud-storage azure-storage-blob

# Run as non-root for cluster security policies
RUN useradd --create-home fiaeu
USER fiaeu

ENTRYPOINT ["fiae"]
CMD ["--help"]
