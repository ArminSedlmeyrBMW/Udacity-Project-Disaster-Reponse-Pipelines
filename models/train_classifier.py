import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import pickle
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, LogisticRegression #recording to this cheat-sheet i should use this https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
from sklearn.tree import DecisionTreeClassifier 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    #1. Load data from database.
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name=database_filepath, con=engine)
    X = df.message.values
    Y = df.drop(columns=['genre', 'id', 'original', 'message']).values    
    return X, Y#, category_names


def tokenize(text):
    #2. Write a tokenization function to process your text data
    #normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenize
    words = word_tokenize(text)
    #remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    #stem/lemmatize
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return words


def build_model(X, Y):
    #3. Build a machine learning pipeline
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    #4. Train pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    category_names = np.unique(y_test)    
    return model, X_train, X_test, y_train, y_test, category_names


def evaluate_model(model, X_test, y_test, category_names):
    #5. Test your model
    y_pred = model.predict(X_test)
    dict_multioutput_prediction_scores = {}
    dict_multioutput_prediction_scores['Precision'] = []
    dict_multioutput_prediction_scores['Accuracy'] = []
    dict_multioutput_prediction_scores['F1'] = []
    dict_multioutput_prediction_scores['Recall'] = []
    for i in np.arange(y_pred.shape[1]):   
        dict_multioutput_prediction_scores['Precision'].append(precision_score(y_test[:,i], y_pred[:,i], labels=category_names, \
                                                                               average='micro'))
        dict_multioutput_prediction_scores['Accuracy'].append(accuracy_score(y_test[:,i], y_pred[:,i]))
        dict_multioutput_prediction_scores['F1'].append(f1_score(y_test[:,i], y_pred[:,i], labels=category_names, average='micro'))
        dict_multioutput_prediction_scores['Recall'].append(recall_score(y_test[:,i], y_pred[:,i], labels=category_names, \
                                                                         average='micro'))
        df_multioutput_prediction_scores=pd.DataFrame(dict_multioutput_prediction_scores)
    print(df_multioutput_prediction_scores.mean())   
    return df_multioutput_prediction_scores


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model, X_train, X_test, y_train, y_test, category_names = build_model(X, Y)
        #model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()