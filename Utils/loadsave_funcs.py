import pandas as pd
import yaml
import os
from warnings import warn
from termcolor import colored
from Utils.Setup_funcs import create_database


def save_data(savelogpath, load_name, save_name, object=None, name_modifier=''):
    """ saves an object (the database) to file. If the object is not a dataframe, turns it into one"""
    save_name = os.path.join(savelogpath, save_name)

    # Avoid overwriting - if os.path.isfile(save_name): ...
    if not isinstance(object, pd.DataFrame):
        indexes = object.__dict__.keys()
        object = pd.DataFrame([x for x in object.__dict__.values()], index=indexes)

    import dill as pickle
    with open(save_name+name_modifier, "wb") as dill_file:
        pickle.dump(object, dill_file)

    print(colored('Database saved as {}'.format(save_name+name_modifier), 'yellow'))



def load_data(savelogpath, load_name, loadas='.pkl'):
    """ load data into a pandas datafrrame"""
    print(colored('Loading database: {}'.format(load_name),color='yellow'))
    db = pd.read_pickle(os.path.join(savelogpath, load_name))
    return db



def load_yaml(fpath):
    """ load settings from a yaml file and return them as a dictionary """
    with open(fpath, 'r') as f:
        settings = yaml.load(f)
    return settings


def load_paths():
    """ load PATHS.yml to set all the user-specific paths correctly """
    filename = './PATHS.yml'
    return load_yaml(filename)

def setup_database(save_folder, save_name, load_name, excel_path, load_database, update_database):
    """
    Creating a new database from scratch using experiments.csv or load a database from a pre-existing file
    """
    if load_database:
        # Load existing database
        db = load_data(save_folder, load_name)

        # Add new sessions from experiments.csv
        if update_database:
            db = create_database(excel_path, database=db)

    # Create new database
    else:
        db = create_database(excel_path)

    save_data(save_folder, load_name, save_name, object=db, name_modifier='')
    save_data(save_folder, load_name, save_name, object=db, name_modifier='_backup')

    return db


def print_plans(load_database, load_name, selector_type, selector):
    """ When starting a new run, print the options specified in Config.py for the user to check """
    if load_database:
        print(colored('\nLoading database: {}'.format(load_name),'blue'))
    else:
        print(colored('\nCreating new database: {}'.format(load_name), 'blue'))
    if selector_type == 'all':
        print(colored('Analyzing all sessions', 'blue'))
    else:
        print(colored('Selector type: {}\nSelector: {}'.format(selector_type, selector), 'blue'))
