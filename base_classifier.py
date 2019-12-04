# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from datetime import datetime

from data_loader import DataLoader


class BaseClassifier(ABC):
    '''
    Base class for all classifier class.
    In order to be used as a part of ensembles classifier,
    each classifier must implement the following interfaces:
    - model: stores the classification model, trained using training datasets of FashionMNIST.
    - train_data and train_labels: training datasets of FashionMNIST.
    - test_data and test_labels: test datasets of FashionMNIST.
    - load_data(): downloads FashionMNIST dataset, divides it into training and test datasets,
    and performs pre-processing (if needed).
    - predict_proba(data): returns a (n, 10) matrix, 
    which holds the probability that each item in input data belongs to each of the 10 categories.
    - test_acc(): returns a dictionary, which must have a key called 'test_acc',
    storing the accuracy when testing this classifier on the test datasets of FashionMNIST.
    '''

    def __init__(self, load_data=True):
        '''
        If a pre-trained model already exists, load_data flag can be set to False
        so that FashionMNIST datasets won't be download when classifier is initialized.
        Note: if FashionMNIST datasets is not downloaded, fit_ensembles_weight method of
        ensembles_classifier will not work.
        '''

        if load_data:
            self.load_data()
        else:
            self.train_data, self.train_labels, self.test_data, self.test_labels = \
                None, None, None, None

        self.model = None

    @abstractmethod
    def format_data(self, data):
        '''
        Different machine learning/deep learning algorithm has different input format.
        This method is used to convert test data of FashionMNIST datasets into the right format
        before performing classification.
        '''

        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def test_acc(self):
        '''
        Returns a dictionary, which must have a key called 'test_acc',
        storing the accuracy when testing this classifier on the test datasets of FashionMNIST.
        '''

        pass

    @abstractmethod
    def predict_proba(self, data):
        '''
        Returns a (n, 10) matrix, which holds the probability that each item
        in input data belongs to each of the 10 categories.
        '''

        pass

    @abstractmethod
    def predict(self, data):
        '''
        Returns an array with size n, holding the labels with highest probability
        of each item in input data.
        '''

        pass

    @abstractmethod

    def load_data(self):
        '''
        To make sure that FashionMNIST datasets is only downloaded once,
        all classifiers will get training and test data from a data loader
        instead of downloading the data themself.
        '''

        self.train_data = DataLoader().train_data
        self.train_labels = DataLoader().train_labels
        self.test_data = DataLoader().test_data
        self.test_labels = DataLoader().test_labels

    def get_class_name(self):
        '''
        Returns the class name to use as indentifier 
        when saving training history,...
        '''

        return self.__class__.__name__

    def get_time_stamp(self):
        '''
        Gets timestamp to use as indentifier 
        when saving training history,...
        '''

        return datetime.now().strftime('%Y%m%d%H%M%S')
