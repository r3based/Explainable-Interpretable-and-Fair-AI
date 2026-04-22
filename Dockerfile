FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision timm \
    scikit-image scikit-learn \
    matplotlib numpy pillow

COPY . .

ENV PYTHONPATH=/app
