FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Danach erst der Quellcode, damit der obige Layer gecacht werden kann
COPY . .

ENTRYPOINT ["python", "client.py"]