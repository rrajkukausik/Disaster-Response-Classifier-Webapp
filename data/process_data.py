#Author -> Raj Kumar kausik
import sys
import pandas as pd
import numpy as np
import nltk 
from sqlalchemy import create_engine
import sqlite3


def load_data(mes_filepath, cat_filepath):
    '''
    This Function reads two csv files and merges them into one big data frame
    input: (
        mes_filepath: csv file 
        cat_filepath: csv file 
            )
    output: (pandas dataframe)
    '''
    # load messages dataset
    mes = pd.read_csv(mes_filepath)
    # load categories dataset
    cat = pd.read_csv(cat_filepath)
    # merge datasets
    df = mes.merge(cat, on='id')
    # return df 
    return df


def clean_data(df):
    '''
    This Function reads an pandas dataframe(df) and formats data for the  ML model
    input: (
        df: pandas dataframe 
            )
     
    output: (pandas dataframe)
    '''
    # create a dataframe(df) of the 36 individual category columns
    cat = df.categories.str.split(';', expand=True)
    # stores the first row
    row = cat[:1]
    # extracting a list of new column names for categories.
    cat_colnames = row.apply(lambda x: x.str.split('-')[0][0], axis=0)
    # renames all  the columns of `categories`
    cat.columns = cat_colnames

    # convert category values to just numbers 0 or 1
    for col in cat:
        # set each value to be the last character of the string
        cat[col] = cat[col].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1)
        # convert col from string to numeric
        cat[col] = cat[col].astype(int)

    # dropping the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenates the original dataframe with the new `categories` dataframe
    df = pd.concat((df, cat), axis=1)
    # drops the  duplicates in the dataframe
    df.drop_duplicates(inplace=True)
    # check number of duplicates in the dataframe
    assert len(df[df.duplicated()]) == 0
    # return the df
    return df


def save_data(df, db_file):
    '''
    The Function saves  the pandas dataframe to the database
    input: (
        df: pandas dataframe 
        db_file: database filename
            )
    output: (No Output)
    '''
    # Giving a name to the table we want to create
    table_name = 'disaster'
    # creating an  db engine 
    engine = create_engine('sqlite:///{}'.format(db_file))
    # Now save dataframe to  Newly created database and replace if already exists  
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
