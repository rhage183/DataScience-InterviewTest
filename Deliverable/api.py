import pandas as pd
from fastapi import FastAPI, Request
from model import get_test_data, get_predictions

app = FastAPI(debug = True)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
def predict():

    X_test, uuids = get_test_data("/home/ralph/code/rhage183/DataScience-InterviewTest/dataset.csv")
    y_pred = get_predictions(X_test, "/home/ralph/code/rhage183/DataScience-InterviewTest/Deliverable/models/model.pkl")

    return {"ID" : uuids.tolist(),
            'PROB' : y_pred.tolist()}


if __name__ == "__main__":
    print(predict())
