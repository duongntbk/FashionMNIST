# -*- coding: utf-8 -*-

import pickle

from sklearn.ensemble import RandomForestClassifier

from base_shallow_classifier import BaseShallowClassifier


class RFClassifier(BaseShallowClassifier):
    '''
    Image classification using random forest classifier (RFC).
    Can reach 87.82% accuracy on test set of FashionMNIST datasets
    using the following parameters:
    - n_estimators=160
    - min_samples_split=2

    Note: actual accuracy may vary based on intial seed.
    '''

    def __init__(self, load_data=True):
        '''
        Simply calls parent's constructor, 
        which in turn calls load_data method (if needed).
        '''

        super().__init__(load_data)

    def get_algorithm(self):
        '''
        Returns the algorithm in use (which is RFC),
        this method is used in cross_validation method.
        '''
        return RandomForestClassifier()

    def train_model(self, save_path, max_obs=None,
            n_estimators=10, min_samples_split=2):
        '''
        Trains the model on training set of FashionMNIST datasets,
        using RFC algorithm. n_estimators and min_samples_split 
        can be set from parameters.
        '''

        if self.train_data is None or self.train_labels is None:
            raise ValueError('Fashion MNIST datasets is not loaded')

        last_train_index = max_obs if max_obs else self.train_data.shape[0]
        train_data = self.train_data[:last_train_index]
        train_labels = self.train_labels[:last_train_index]

        self.model = RandomForestClassifier(n_estimators=n_estimators,
            min_samples_split=min_samples_split)
        self.model.fit(train_data, train_labels)

        with open(save_path, 'wb') as f:
            f.write(pickle.dumps(self.model))
