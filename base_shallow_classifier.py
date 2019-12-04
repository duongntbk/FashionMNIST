# -*- coding: utf-8 -*-

import pickle
from abc import abstractmethod

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from base_classifier import BaseClassifier


class BaseShallowClassifier(BaseClassifier):
    '''
    Base class of classifier using traditional machine learning algorithm.
    '''

    def __init__(self, load_data=True):
        super().__init__(load_data)

    @abstractmethod
    def get_algorithm(self):
        '''
        Returns the algorithm in use, this will be used inside cross_validation method
        as a parameter of GridSearch.
        '''

        pass

    def load_data(self):
        '''
        Copies FashionMNIST datasets from data loader object,
        then performs normalization and one-hot encoding.
        '''

        super().load_data()

        self.label_encoder = None

        # Normalization, most machine learning techniques 
        # works best when data value is close to 1
        self.train_data = self.train_data.astype('float32') / 255
        self.train_data = self.train_data.reshape(self.train_data.shape[0], 28*28)
        self.test_data = self.test_data.astype('float32') / 255
        self.test_data = self.test_data.reshape(self.test_data.shape[0], 28*28)
        
        # One-hot encoding
        self.label_encoder = LabelEncoder()
        self.train_labels = self.label_encoder.fit_transform(self.train_labels)
        self.test_labels = self.label_encoder.fit_transform(self.test_labels)

    def cross_validation(self, param_grid, cv, max_obs=None):
        '''
        Uses cross validation and grid search to find the best parameters.
        - param_grid: all values of parameters to be tested.
        - cv: the k value in k-th fold cross validation.
        - max_obs: the maximum number of observations to search on.
          (if use the full 50000 training data observations,
          some algorithms will take a very long time).
        Note: this method only uses the training set, test set is not used.
        '''

        if self.train_data is None or self.train_labels is None:
            raise ValueError('Fashion MNIST datasets is not loaded') 

        last_train_index = max_obs if max_obs else self.train_data.shape[0]
        train_data = self.train_data[:last_train_index]
        train_labels = self.train_labels[:last_train_index]

        grid_search = GridSearchCV(self.get_algorithm(), param_grid, cv=cv)
        grid_search.fit(train_data, train_labels)

        print('Test set score: {:.2f}'.format(grid_search.score(train_data, train_labels)))
        print('Best parameters: {}'.format(grid_search.best_params_))
        print('Best cross-validation score: {:.2f}'.format(grid_search.best_score_))

    def test_acc(self):
        '''
        Tests pre-trained model using test set of FashionMNIST datasets.
        '''

        if self.test_data is None or self.test_labels is None:
            raise ValueError('Fashion MNIST datasets is not loaded') 
    
        if not self.model:
            raise ValueError('Please load a model before performing accuracy testing.')

        pred = self.model.predict(self.test_data)
        acc = accuracy_score(pred, self.test_labels)

        return {'test_acc':acc}

    def load_model(self, model_path):
        '''
        Loads pre-trained model from storage.
        '''

        with open(model_path, 'rb') as f:
            self.model = pickle.loads(f.read())

    def format_data(self, data):
        '''
        All shallow learning algorithms in this project takes input
        with format (n, 784).
        '''

        if data.shape[:1] != (28*28,):
            data = data.reshape(data.shape[0], 28*28)
        return data

    def predict_proba(self, data):
        '''
        Returns a (n, 10) matrix, which holds the probability that each item
        in input data belongs to each of the 10 categories.
        '''

        if not self.model:
            raise ValueError('Please load a model before performing classification.')

        data = self.format_data(data)
        pred = self.model.predict_proba(data)
        return pred
    
    def predict(self, data):
        '''
        Returns an array with size n, holding the labels with highest probability
        of each item in input data.
        '''

        if not self.model:
            raise ValueError('Please load a model before performing classification.')

        data = self.format_data(data)
        pred = self.model.predict(data)
        return pred
