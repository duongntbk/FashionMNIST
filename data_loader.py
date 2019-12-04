# -*- coding: utf-8 -*-

from keras.datasets import fashion_mnist

class DataLoader:
    '''
    Helper class to load FashionMNIST datasets.
    This class is a singleton.
    '''

    __is_create = False

    def __init__(self):
        '''
        This class can only be initialized once.
        '''

        # Only initialize this class if __is_create flag is False.
        if not DataLoader.__is_create:
            (DataLoader.train_data, DataLoader.train_labels), \
                (DataLoader.test_data, DataLoader.test_labels) = fashion_mnist.load_data()
            DataLoader.__is_create = True # set __is_create to True after class initialization.
