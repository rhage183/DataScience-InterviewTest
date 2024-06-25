FROM python:3.10.6-buster

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY Deliverable Deliverable
COPY dataset.csv dataset.csv
CMD uvicorn api:app --host 0.0.0.0
