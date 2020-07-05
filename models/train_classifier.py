import sys
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(db_file):
    '''
    This Function Loads data from the sqlite database we created in the process_data.py python file
    input: (
        db_file: path to database
            ) 
    output: (
        X: features dataframe
        y: target dataframe
        cat_names: names of targets
        )
    '''
    # table name
    table_name = 'disaster'
    # loads data from database
    engine = create_engine('sqlite:///{}'.format(db_file))
    df = pd.read_sql_table(table_name, engine)
    # create X and y values
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1)
    # get the col names 
    cat_names = y.columns
    return X, y, cat_names


def tokenize(text, lemma=True, rm_stop=True, rm_short=True):
    '''
    This Function tokenizes the text into Simple tokens when a text is given to it
    input: (
        text: 
        lemma: True if lemmatize word, 
        rm_stop: True if remove stopwords  
        rm_short: True if remove short words > 2
            ) 
    output: (
        It returns cleaned tokens in form of list 
            )
    '''
    # Store the stopword list of english language 
    STOPWORDS = list(set(stopwords.words('english')))
    # initializes the  lemmatier
    lemmatizer = WordNetLemmatizer()
    # This splits string into words (tokens)
    tokens = word_tokenize(text)
    # Now remove the  short words
    if rm_short: tokens = [t for t in tokens if len(t) > 2]
    # Now put words into base form i.e, lemmatized form
    if lemma: tokens = [lemmatizer.lemmatize(t).lower().strip() for t in tokens]
    # This removes all the  stopwords
    if rm_stop: tokens = [t for t in tokens if t not in STOPWORDS]
    # return the cleaned tokens
    return tokens


def build_model():
    '''This Function Builds classification model by using the pipeline'''
    # Creating the Model Pipeline 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=100)))
    ])
    # Setting the hyper-parameter grid for the model
    param_grid = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': ['log2', 'auto'],
        'clf__estimator__n_estimators': [100, 250],
    }
   
    # Instantiating the Grid search cv 
    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=2, n_jobs=4, cv=3)
    # returns the best model
    return cv


def evaluate_model(model, X_test, Y_test, cat_names):
    '''
    This Function Evaluates the trained model against the test dataset
    input: (
        model: trained model 
        X_test: Test features 
        Y_test: Test labels 
        cat_names: names of lables
            )
    '''
    #  predicting in the Test set 
    y_preds = model.predict(X_test)
    # printing the  classification report
    print(classification_report(y_preds, Y_test.values, target_names=cat_names))
    # printing the  raw accuracy score of the model
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_preds)))


def save_model(model, mod_file):
    '''
    This Function Saves the model into a Python pickle file (.pkl extension)
    input: (
        model: trained model 
        mod_file: filepath to save model in binary form 
            )
    '''
    #It saves model binary in the given path 
    pickle.dump(model, open(mod_file, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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