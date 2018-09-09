import pandas as pd
import yaml
import os
from warnings import warn


def save_data(savelogpath, save_name, object=None, name_modifier='', saveas='pkl'):
    """ saves an object (the database) to file. If the object is not a dataframe, turns it into one"""
    try:
        save_name = os.path.join(savelogpath, save_name)
        if not isinstance(object, pd.DataFrame):
            indexes = object.__dict__.keys()
            object = pd.DataFrame([x for x in object.__dict__.values()], index=indexes)

        if saveas == 'pkl':
            import dill as pickle
            with open(save_name, "wb") as dill_file:
                pickle.dump(object, dill_file)
        else:
            # TODO doesnt work
            pass
            # object.to_hdf(save_name, key='df', mode='a')
            # """ https://glowingpython.blogspot.com/2014/08/quick-hdf5-with-pandas.html """
            # store = pd.HDFStore(save_name)
            # store.put('data', object, format='table', data_columns=True)

        print('           ... data saved as {}'.format(save_name))
    except:
        if object is None:
            warn('           ... tried to save a "None" object')
        else:
            warn('           ... something went wrong with saving')


def load_data(savelogpath, load_name, loadas='.pkl'):
    """ load data into a pandas datafrrame"""
    print('====================================\n====================================\n'
          'Loading database from: {}'.format(load_name))
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


