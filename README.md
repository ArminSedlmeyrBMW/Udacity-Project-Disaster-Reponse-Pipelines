# Disaster Response Pipeline Project
This README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository.
link to github repo: https://github.com/ArminSedlmeyrBMW/Udacity-Project-Write-a-Data-Science-Blog-Post

### Project Motivation
I am a Udacity student who is doing the project Disaster Reponse Pipelines within this repo.

### Summary of the project
1. This project takes social media posts as an input within known rubrics
2. Then the data is wrangled and saved to a sql-lite database
3. Then the data is used to train an ML-model
4. Then a web-app is deployed with takes inputs from users as a text
5. The ML-model uses the input to label the text and outputs it back to the web app

### Explanation of the files in the repository
- classifier.pkl --> the machine-learning-model as a pickle file
- train_classifier.py --> the code which builds the ML-model by loading the data from the sql-lite database
- process_data.py --> the code to load, wrangle and save the data to a sql-lite database
- run.py --> the code to run the web app
- disaster_categories.csv --> labels
- disaster_messages.csv --> input text for creating features
- DisasterResponse.db --> sql-lite-data-base with the wrangled data

### Installations required:
- import json
- import plotly
- import sys
- import pandas as pd
- from sqlalchemy import create_engine
- import re
- import numpy as np
- import nltk

- from nltk.tokenize import word_tokenize
- from nltk.corpus import stopwords
- from nltk.stem import WordNetLemmatizer
- from nltk.stem import PorterStemmer
- from wordcloud import WordCloud

- from flask import Flask
- from flask import render_template, request, jsonify
- from plotly.graph_objs import Bar
- from sklearn.externals import joblib
- from sqlalchemy import create_engine

- nltk.download(['punkt', 'wordnet', 'stopwords'])

- from sklearn.base import BaseEstimator, TransformerMixin
- from sklearn.pipeline import Pipeline, FeatureUnion
- from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, classification_report, accuracy_score
- from sklearn.model_selection import train_test_split, GridSearchCV
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.kernel_approximation import RBFSampler
- from sklearn.linear_model import SGDClassifier, LogisticRegression #recording to this cheat-sheet i should use this https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
- from sklearn.tree import DecisionTreeClassifier 
- from sklearn.multioutput import MultiOutputClassifier
- from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


### Instructions (how to run the Python scripts and web app):
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app.
    python run.py

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/
