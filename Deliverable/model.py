"""
This file contains functions that have been packaged based on the exploration.ipynb notebook
They contain data "preprocessing functions", "model training and saving" functions and "Making prediction" functions
For simplicity they are all kept in a single .py file for now

Contains an if __name__ == '__main__': that runs a preprocessing, trains a model, saves it, runs prediction and finally
writes them into a csv file
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler


import xgboost as xgb
import pickle

def make_preproc_pipeline(X:pd.DataFrame) -> Pipeline:
    """Function that takes the data as input and returns the preprocessing pipeline

    Contains:   Simple Imputer and Robust scaler for numeric data
                One Hot encoder for categorica data
    Args:
        X (pd.Dataframe): Dataframe of data

    Returns:
        Pipeline: Preprocessing Pipeline
    """

    num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),#Median is more robust to outliers, might be safer
    ('scaler', RobustScaler())
    ])

    cat_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=0.05))
    ])

    num_columns = X.select_dtypes(include=(float, int)).columns.tolist()
    cat_columns = X.select_dtypes(exclude=(float, int)).columns.tolist()

    Preproc_pipeline = ColumnTransformer(transformers=[
    ('num', num_pipe, num_columns),
    ('cat', cat_pipe, cat_columns)
    ])

    return Preproc_pipeline

def make_full_pipeline(prep_pipeline : Pipeline) -> Pipeline:
    """ Function that takes a preprocessing pipeline as input and returns the full
    pipeline (Adding estimator), Use case if XGBClassifier

    Args:
        Pipeline (Pipeline): Preprocessing pipeline

    Returns:
        Pipeline: Full pipeline
    """
    return Pipeline([
    ('preproc', prep_pipeline),
    ('estimator',xgb.XGBClassifier(max_depth = 5,
                                   n_estimators = 10,
                                   n_jobs = -1))
    ])

def preprocess_bef_train() -> tuple[np.array, np.array]:
    """ Preprocessing of data before training, in case of required training of model

    Args:
        df (_type_): Initial dataframe

    Returns:
        _type_: tuple of preprocessed features and target that have been undersampled
    """
    #Load data
    df = pd.read_csv("dataset.csv", delimiter=';')

    #First preprocessing required
    df = df.dropna(axis = 'columns', thresh=int(df.shape[0]*0.6)).dropna(axis = 0, subset="default")
    df['has_paid'] = df['has_paid'].astype("int64")

    y = df['default']
    X = df.drop(columns=['uuid','default'])

    X_rus , y_rus = RandomUnderSampler(random_state=42).fit_resample(X,y)

    return X_rus, y_rus

def get_test_data(PATH) -> pd.DataFrame:
    """ Function that extracts the test data required for the exercise

    Returns:
       X_test : A pandas dataframe of the unprocessed features for the test prediction
       uuids : uuids of prediction targets
    """
    data = pd.read_csv(PATH, delimiter=';')
    uuids = np.array(data.iloc[-10000:,:]['uuid']) # Gets the last 10K uuids in dataset corresponding to the ids that have to be returned for the test

    test_data = data.iloc[-10000:,:]

    X_test = test_data.drop(columns=['uuid','default'])

    return X_test, uuids

def train_model(full_pipeline:ColumnTransformer, features : pd.DataFrame, target : pd.Series,
                show_metrics:bool = True, save_model:bool = True):
    """ Function that trains an estimator, shows its training metrics and saves it

    Args:
        full_pipeline (ColumnTransformer): Full pipeline including preprocessor and estimator
        features (pd.DataFrame): Features of dataset: X
        target (pd.Series): target of dataset: y
        show_metrics (bool, optional): controls wether metrics and confusion matrix are shown. Defaults to True.
        save_model (bool, optional): controls wether model is saved at the end of training. Defaults to True.
    """

    if 'uuid' in features.columns:
        features = features.drop(columns=['uuid'])
    if 'default' in features.columns:
        features = features.drop(columns=['default'])

    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, stratify=target)

    full_pipeline = full_pipeline.fit(X_train, y_train)

    if show_metrics:
        #Scoring Model
        predictions = full_pipeline.predict(X_val)
        print("f1 Score is: ", f1_score(y_val,predictions))

        #Showing Confusion matrix
        ConfusionMatrixDisplay(confusion_matrix(y_val, predictions), display_labels=("Did not default", "Default")).plot()
        plt.savefig("Confusion_Matrix.jpg")
        print("Saved Confusion Matrix")

    #Saving model
    if save_model:
        pickle.dump(full_pipeline, open('Deliverable/models/model.pkl', 'wb'))
        print("Saved model")

def get_predictions(X_test:pd.DataFrame, PATH: str) -> np.array:
    """Functions that takes test data as input and returns the model predictions

    Args:
        X_test (_type_): Dataframe of test data

    Returns:
        y_pred: returns the predicted probabilities of default
    """
    # Load trained model
    tr_model = pickle.load(open(PATH, 'rb'))

    #Gets rid of uuid and default columns in case they still exist
    if 'uuid' in pd.DataFrame(X_test).columns:
        X_test = X_test.drop(columns=['uuid'])
    if 'default' in pd.DataFrame(X_test).columns:
        X_test = X_test.drop(columns=['default'])

    #Make predictions
    y_pred = tr_model.predict_proba(X_test)[:,1]

    return np.array(y_pred)

def get_prediction_file(uuids:np.array, y_pred:np.array):
    """Creates a csv file containing the uuids and the predicted probability of default

    Args:
        uuids (np.array): prediction uuids
        y_pred (np.array): prediction probablities
    """

    pd.DataFrame({"uuid" : uuids,
                  "pd" : y_pred}).to_csv("predictions.csv", index=False)
    print("Saved prediction to csv")

if __name__ == '__main__':

    print("Getting Test Data")
    X_test, uuids = get_test_data("/home/ralph/code/rhage183/DataScience-InterviewTest/dataset.csv")
    print("X_test and uuids shape are", (X_test.shape,uuids.shape))

    print('First step of preprocessing data for training')
    X_train, y_train = preprocess_bef_train()
    print('Using random UnderSampling to balance target classes')
    print("X_train and y_train shape are", (X_train.shape,y_train.shape))

    print("Preprocessing training data")
    X_train_prepro = make_preproc_pipeline(X_train).fit_transform(X_train)

    print('Getting full pipeline')
    full_pipe = make_full_pipeline(make_preproc_pipeline(X_train))

    print("Training Model:")
    train_model(full_pipeline=full_pipe, features=X_train, target=y_train, show_metrics=True, save_model=True)

    print('Getting predictions')
    y_pred = get_predictions(X_test, PATH="/home/ralph/code/rhage183/DataScience-InterviewTest/Deliverable/models/model.pkl")

    print('Saving predictions')
    get_prediction_file(uuids=uuids, y_pred=y_pred)
