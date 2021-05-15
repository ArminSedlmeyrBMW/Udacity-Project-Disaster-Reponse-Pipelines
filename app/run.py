import json
import plotly
import pandas as pd
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    #normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenize
    words = word_tokenize(text)
    #remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    #stem/lemmatize
    words = [PorterStemmer().stem(w) for w in words]
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    return clean_tokens

# load data
#database_filepath = sys.argv[1:]
#engine = create_engine('sqlite:///../{}'.format(database_filepath))
#df = pd.read_sql_table(database_filepath, engine)
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# create and save picture of word-cloud, since plotly does not have "word-cloud" yet for displaying it as "interactive" plot
if 0: #@reviewer: hi, due to performance-reasons I commented this out and uploaded the picture to the workspace manually after computing it in the Workspace - ETL - section 9 if you want to double check it works, just run it there.
    str_messages=' '.join(df.message)
    tokenized_messages=tokenize(str_messages)
    str_=' '.join(tokenized_messages)
    wordcloud = WordCloud(width = 1000, height = 700).generate(str_)
    plt.figure(figsize=(18,12))
    plt.axis('off')
    plt.imshow(wordcloud);
    plt.savefig('\static\img\WordCloud-Message-Training-Data.png')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    x = df.drop(columns=['id', 'message', 'original', 'genre']).corr().related.index.values
    y = df.drop(columns=['id', 'message', 'original', 'genre']).corr().related.index.values
    z = np.where(np.tril(df.drop(columns=['id', 'message', 'original', 'genre']).corr().values)==1, 0, np.tril(df.drop(columns=['id', 'message', 'original', 'genre']).corr().values))
    
    # create visuals
    graphs = [
        { # correlation matrix
            'data': [
                {
                "type": 'heatmap',
                "x": x,
                "y": y,
                "z": z
                }
            ],

            'layout': {
                'height': 1000,
                'width': 1000,
                'title': 'Correlation Matrix of Output Labels',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        { # udacity example - genre counts
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()