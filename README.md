# Disaster Response Pipeline Project
This README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository.

### Summary of the project
1. This project takes social media posts as an input within known rubrics
2. Then the data is wrangled and saved to a sql-lite database
3. Then the data is used to train an ML-model
4. Then a web-app is deployed with takes inputs from users as a text
5. The ML-model uses the input to label the text and outputs it back to the web app

### Explanation of the files in the repository
classifier.pkl --> the machine-learning-model as a pickle file
train_classifier.py --> the code which builds the ML-model by loading the data from the sql-lite database
process_data.py --> the code to load, wrangle and save the data to a sql-lite database
run.py --> the code to run the web app and execute relevant

### Instructions (how to run the Python scripts and web app):
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/
