# FROM python:3.13-alpine
# COPY . /app
# WORKDIR /app
# RUN pip install -r requirements.txt
# CMD python app.py
FROM python:3.13-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# CMD ["python", "app.py"]
