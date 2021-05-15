
import re
import os
import joblib
import argparse
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from azureml.core.run import Run
from datetime import datetime

run = Run.get_context()

def clean_df(ds):
    # Select only HQ and LQ_CLOSE
    ds = ds[ds['Y'].isin(['HQ', 'LQ_CLOSE'])].copy()
    # Concatenate title and body
    ds['text'] = ds['Title'] + ' ' + ds['Body']
    # Clean the target array
    ds['y'] = ds['Y'].apply(lambda x: 1 if x == 'HQ' else 0)
    return ds[['text', 'y']].copy()


def clean_text(raw_text):
    # Remove codes
    text = re.sub(r"<code>(.*?)</code>", "", raw_text, flags=re.DOTALL)
    # Remove html tags
    text = re.sub(r"<.*?>", "", text)
    # Remove new lines
    text = re.sub(r"\n", '', text)
    # Remove links
    text = re.sub(r"http\S+", "", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]"," ",text)
    # State a flag on all digits
    text = re.sub(r"\d+", r"\\d", text)
    text = re.sub(r"\\d", "_number_", text)
    text = text.lower()
    return text


def preprocess_text(text):
    # Set the minimum frequency to curb dimensionality
    vectorizer = CountVectorizer(min_df=0.025)
    X = vectorizer.fit_transform(text)
    return X.toarray()


def main():
    dataset = pd.read_csv('Data/train.csv')
    dataset = clean_df(dataset)
    dataset['text'] = dataset['text'].apply(clean_text)

    X = preprocess_text(dataset['text'].tolist())
    y = dataset['y'].to_numpy()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    
    # import libraries, save the model
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning Rate")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of Estimators")


    args = parser.parse_args()

    run.log("Learning Rate", float(args.learning_rate))
    run.log("Number of Estimators", int(args.n_estimators))


    # specify your configurations as a dict
    params = {
        'validation_fraction': 0.2,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators
    }

    # train
    gbm = GradientBoostingClassifier(**params).fit(X_trainval, y_trainval)

    # save the model after the run has completed.
    learning_rate = round(args.learning_rate, 4)
    n_estimators = args.n_estimators

    timestamp = datetime.now(tz=None).strftime("%Y-%m-%d %H-%M")
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(gbm, f'outputs/hd_model {timestamp} -lr={learning_rate} -ne={n_estimators}.joblib')

    # predict
    y_pred = gbm.predict(X_test)
    y_pred_1_0 = (y_pred > 0.5).astype(int)
    # eval
    auc_score = roc_auc_score(y_test, y_pred_1_0)
    accuracy = accuracy_score(y_test, y_pred_1_0)
    run.log("AUC", float(auc_score))
    run.log("Accuracy", float(accuracy))



if __name__ == '__main__':
    main()