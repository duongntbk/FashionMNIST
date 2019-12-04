# -*- coding: utf-8 -*-

import numpy as np

class EnsemblesClassifier:
    '''
    This class pools together the predictions of a set of different classifiers,
    to produce better predictions.
    Each classifier looks at slightly different aspects of the data to make its predictions, 
    getting part of the truth but not all of it.
    As the old saying goes, The Whole is Greater than the Sum of its Parts.

    An accuracy of 92.41% can be reached with the following set of classifiers:
    - CNN classifier with 92.01% accuracy, 0.4218 weight.
    - DNN classifier with 89.51% accuracy, 0.2447 weight.
    - SVC classifier with 86.69% accuracy, 0.1520 weight.
    - RFC classifier with 87.82% accuracy, 0.1815 weight.
    '''

    def __init__(self, classifier_list=None):
        '''
        Classifier list may be set at initialization or later.
        '''

        if classifier_list:
            self.classifier_list = classifier_list
        else:
            self.classifier_list = []

        self.ensembles_weight_list = None
    
    def set_ensembles_weight(self, weights):
        '''
        Populates the ensembles weight list.
        '''

        if len(weights) != len(self.classifier_list):
            raise ValueError('Lenght of list of ensembles weight must match len of classifier list')

        self.ensembles_weight_list = weights

    def fit_ensembles_weight(self):
        '''
        Let the accuracy of model i be Xi,
        Then model ensembling works best when the weight of each model is proportional to
        (1-Xi)^-2.
        The better classifiers are given a higher weight,
        and the worse classifiers are given a lower weight.
        '''

        for classifier in self.classifier_list:
            if classifier.test_data is None or classifier.test_labels is None:
                classifier.load_data()
        
        acc_list = np.array([classifier.test_acc()['test_acc'] for classifier in self.classifier_list])
        inverse_delta_square = (1 - acc_list)**-2
        total_inverse_delta_square = np.sum(inverse_delta_square)
        self.ensembles_weight_list = inverse_delta_square / total_inverse_delta_square

    def predict_precheck(self):
        '''
        Checks if both classifier list and weight list are not empty.
        Also, those 2 list must have the same length.
        '''

        if self.classifier_list is None or self.ensembles_weight_list is None or \
                len(self.classifier_list) != len(self.ensembles_weight_list):
            return False
        
        for classifier in self.classifier_list:
            if not classifier.model:
                return False

        return True

    def predict_proba(self, data):
        '''
        Returns a (n, 10) matrix, which holds the probability that each item
        in input data belongs to each of the 10 categories.
        '''

        if not self.predict_precheck():
            raise AttributeError('Classifier list or ensembles weight list is not valid')

        pred_list = [classifier.predict_proba(data) * weight for classifier, weight in \
                        zip(self.classifier_list, self.ensembles_weight_list)]

        return np.sum(pred_list, axis=0)

    def predict(self, data):
        '''
        Returns an array with size n, holding the labels with highest probability
        of each item in input data.
        '''

        pred_proba = self.predict_proba(data)
        return np.argmax(pred_proba, axis=1)
