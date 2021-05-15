import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """load and merge .csv files to pd-DataFrame
    Args:
        messages_filepath: string. path to the messages .csv file
        categories_filepath: string. path to the categories .csv file
    Returns:
        df: pandas DataFrame
        """
    
    #1. Load datasets.
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #2. Merge datasets.
    df = categories.merge(on='id', right=messages)
    return df


def clean_data(df):
    """ splits the categories column into separate, clearly named columns, converts values to binary, and drops duplicates.
    Args:
        df: pandas DataFrame
    Returns:         
        df: pandas DataFrame
        """    
    #3. Split categories into separate category columns.
    categories = df.categories.str.split(pat=';', expand=True)
    col_names = df.categories.head(1).str.split(pat=';')
    
    i=0
    rename_dict={}
    for col_old in categories.columns.values:
        rename_dict[col_old]=col_names[0][i][0:-2]
        i=i+1
    categories.rename(columns=rename_dict, inplace=True)    
    #4. Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)    
    #5. Replace categories column in df with new category columns.  
    df.drop(columns=['categories'], inplace=True)
    df=pd.concat([df, categories], axis=1)
    # 6. Remove duplicates.
    df.duplicated().sum()
    df.drop_duplicates(inplace=True)    
    return df


def save_data(df, database_filename):
    """ creates sql-lite database
    Args:
        df: pandas DataFrame
        database_filename: name of database
    """       
    #7. Save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False)

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