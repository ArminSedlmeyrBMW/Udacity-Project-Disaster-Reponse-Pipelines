# Disaster Response Pipeline Project
This README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository.
link to github repo: https://github.com/ArminSedlmeyrBMW/Udacity-Project-Write-a-Data-Science-Blog-Post

### Project Motivation
I am a Udacity student who is doing the project Disaster Reponse Pipelines within this repo.

### Summary of the project
[**Why?**](https://www.smartinsights.com/digital-marketing-strategy/online-value-proposition/start-with-why-creating-a-value-proposition-with-the-golden-circle-model/)
The main goal of this projects web application is to help people or organizations during an event of disaster.
**How?**
1. This project takes social media posts as an input within known rubrics
2. Then the data is wrangled and saved to a sql-lite database
3. Then the data is used to train a ML-model
4. Then a web-app is deployed which takes inputs from users as a text
5. The ML-model uses the input to label the text and outputs it back to the web app
**What?**
Reading in social media posts we can classifiy if the user needs help or not, and for what exactly the possible help is needed.

### Explanation of the files in the repository
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
| - static
| | - img
| | | - WorldCloud-Message-Training-Data.png, a picture showing a word-Cloud for the messages (training data)
| | - 
| - run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process, labels
|- disaster_messages.csv # data to process, input text for creating features
|- process_data.py,  the code to load, wrangle and save the data to a sql-lite database
|- DisasterResponse.db # database to save clean data to, sql-lite-data-base with the wrangled data
models
|- train_classifier.py, the code which builds the ML-model by loading the data from the sql-lite database
|- classifier.pkl # saved model, the machine-learning-model as a pickle file
README.md

### Installations required:

- json
- plotly
- sys
- pandas as pd
- sqlalchemy
- re
- numpy
- nltk
- ordcloud import WordCloud
- flask
- plotly
- sklearn

### Instructions (how to run the Python scripts and web app):
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app.
    python run.py

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/
