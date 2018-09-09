import platform
import pandas as pd
import yaml
import os
import dill as pickle


def save_data(savelogpath, save_name, object=None, name_modifier='', saveas='pkl'):
    # Get full file name and save to pickel file
    if platform.system() == 'Windows':
        save_name = '{}\\{}{}.pkl'.format(savelogpath, save_name, name_modifier)
    else:
        save_name = '{}/{}{}.pkl'.format(savelogpath, save_name, name_modifier)
    try:
        with open(save_name, 'ab') as output:
            if isinstance(object, pd.DataFrame):
                if not saveas == 'h5':
                    with open(save_name, "wb") as dill_file:
                        pickle.dump(object, dill_file)

                else:
                    """ https://glowingpython.blogspot.com/2014/08/quick-hdf5-with-pandas.html """
                    store = pd.HDFStore(save_name)
                    store.put('data', object, format='table', data_columns=True)
            else:
                indexes = object.__dict__.keys()
                data_to_save = pd.DataFrame([x for x in object.__dict__.values()], index=indexes)
                if not saveas == 'h5':
                    data_to_save.to_pickle(save_name)
                else:
                    data_to_save.to_hdf(save_name, key='df', mode='a')
            print('           ... data saved succefully ')
    except:
        print('           ... something went wrong with saving')
        """
        I think that this happens when save_data is called twice with very little time in between. If the database
        hasn't been completely saved when this function is called again there will be an error. 
        
        This might happen during tracking if one session has no trials: the db will be saved at the end of a session, 
        then the session with no trials will be processed very quickly and the code will try to save db again. But if
        this happens too quickly the previous save operation hasn't terminated yet. 
        
        If this is the case then we can just ignore this error
        """



def load_data(savelogpath, load_name, loadas='.pkl'):
    print('====================================\n====================================\n'
          'Loading database from: {}'.format(load_name))
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


