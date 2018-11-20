import pandas as pd
import yaml
import os
from warnings import warn
from termcolor import colored


def save_data(savelogpath, load_name, save_name, loaded_db_size, object=None, name_modifier=''):
    """ saves an object (the database) to file. If the object is not a dataframe, turns it into one"""
    save_name = os.path.join(savelogpath, save_name)

    # Avoid overwriting - if os.path.isfile(save_name): ...
    if not isinstance(object, pd.DataFrame):
        indexes = object.__dict__.keys()
        object = pd.DataFrame([x for x in object.__dict__.values()], index=indexes)

    import dill as pickle
    with open(save_name+name_modifier, "wb") as dill_file:
        pickle.dump(object, dill_file)

    counter = 0
    while os.path.getsize(save_name) < loaded_db_size-1:  # wait until saving is done before continuing
        counter += 1
        if counter > 10000:
            break

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


