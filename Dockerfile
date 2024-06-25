FROM python:3.10.6-buster

WORKDIR /proj

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY Deliverable Deliverable
COPY dataset.csv dataset.csv
COPY setup.py setup.py
CMD uvicorn Deliverable.api:app --host 0.0.0.0
