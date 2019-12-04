# -*- coding: utf-8 -*-

import pickle

from sklearn.svm import SVC

from base_shallow_classifier import BaseShallowClassifier


class SVCClassifier(BaseShallowClassifier):
    '''
    Image classification using support vector clustering (SVC).
    Can reach 86.69% accuracy on test set of FashionMNIST datasets
    using the following parameters:
    - max_obs: 10000
    - gamma: 0.01
    - C: 10
    - kernel: 'rbf'

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
        Returns the algorithm in use (which is SVC),
        this method is used in cross_validation method.
        '''

        return SVC()

    def train_model(self, save_path, max_obs=None,
            gamma='auto_deprecated', C=1.0, kernel='linear'):
        '''
        Trains the model on training set of FashionMNIST datasets,
        using SVC algorithm. Gamma, C and kernel can be set from parameters.
        Note: SVC does not scale very well on datasets with more than 
        a few ten-thousands observations, max_obs should not be more than 10000.
        '''

        if self.train_data is None or self.train_labels is None:
            raise ValueError('Fashion MNIST datasets is not loaded') 

        last_train_index = max_obs if max_obs else self.train_data.shape[0]
        train_data = self.train_data[:last_train_index]
        train_labels = self.train_labels[:last_train_index]

        self.model = SVC(gamma=gamma, C=C, kernel=kernel, probability=True)
        self.model.fit(train_data, train_labels)

        with open(save_path, 'wb') as f:
            f.write(pickle.dumps(self.model))
