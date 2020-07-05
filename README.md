# Udacity Data Scientist Nanodegree Project 2

## Project Name - Disaster Response Pipeline Webapp

## Building an WebApp to classify Disaster Response 

### Data
In this project, I have applied my skills to analyze disaster data from **Figure Eight** to build a model for an API that classifies disaster messages.
Link for data - https://appen.com/

This WebApp can solve real world usecase such as helping us classifying data for disaster response.

The disaster response management team can easily get that the message wants to convey , and thus can take steps regarding it to solve the problem.

### Instructions to Run the Webapp:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



