import platform
import pandas as pd
import yaml


def save_data(savelogpath, save_name, object=None, name_modifier=''):
    # Get full file name and save to pickel file
    if platform.system() == 'Windows':
        save_name = '{}\\{}{}.pkl'.format(savelogpath, save_name, name_modifier)
    else:
        save_name = '{}/{}{}.pkl'.format(savelogpath, save_name, name_modifier)
    with open(save_name, 'ab') as output:
        if isinstance(object, pd.DataFrame):
            object.to_pickle(save_name)
        else:
            indexes = object.__dict__.keys()
            data_to_save = pd.DataFrame([x for x in object.__dict__.values()], index=indexes)
            data_to_save.to_pickle(save_name)
    print('Data saved succefully')


def load_data(savelogpath, load_name):
    print('==================\nLoading database from: {}'.format(load_name))
    if platform.system() == 'Windows':
        filename = '{}\\{}.pkl'.format(savelogpath, load_name)
    else:
        filename = '{}/{}.pkl'.format(savelogpath, load_name)

    db = pd.read_pickle(filename)
    return db


def load_yaml(fpath):
    with open(fpath, 'r') as f:
        settings = yaml.load(f)
    return settings


