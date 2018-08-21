import platform
import pandas as pd
import yaml
import os


def save_data(savelogpath, save_name, object=None, name_modifier='', saveas='pkl'):
    # Get full file name and save to pickel file
    if platform.system() == 'Windows':
        save_name = '{}\\{}{}.pkl'.format(savelogpath, save_name, name_modifier)
    else:
        save_name = '{}/{}{}.pkl'.format(savelogpath, save_name, name_modifier)
    with open(save_name, 'ab') as output:
        if isinstance(object, pd.DataFrame):
            if not saveas == 'h5':
                object.to_pickle(save_name)
            else:
                object.to_hdf(save_name, key='df', mode='a')
        else:
            indexes = object.__dict__.keys()
            data_to_save = pd.DataFrame([x for x in object.__dict__.values()], index=indexes)
            if not saveas == 'h5':
                object.to_pickle(save_name)
            else:
                object.to_hdf(save_name, key='df', mode='a')

    print('         ... data saved succefully ')


def load_data(savelogpath, load_name, loadas='.pkl'):
    print('==================\nLoading database from: {}'.format(load_name))
    filename = os.path.join(savelogpath, load_name+loadas)
    db = pd.read_pickle(filename)
    return db


def load_yaml(fpath):
    with open(fpath, 'r') as f:
        settings = yaml.load(f)
    return settings


def load_paths():
    """ load PATHS.yml to set all the user-specific paths correctly"""
    filename = './PATHS.yml'
    return load_yaml(filename)


