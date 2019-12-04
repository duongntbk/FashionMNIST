# -*- coding: utf-8 -*-

from keras.layers import Dense
from keras.models import Sequential

from base_ann_classifier import BaseANNClassifier


class DNNClassifier(BaseANNClassifier):
    '''
    Image classification using dense neural network (DNN).
    Can reach 89.51% accuracy on test set of FashionMNIST datasets.

    Note: actual accuracy may vary based on intial seed.
    '''

    def __init__(self, load_data=True):
        '''
        Simply calls parent's constructor, 
        which in turn calls load_data method (if needed).
        '''

        super().__init__(load_data)

    def load_data(self):
        '''
        If load_data flag is True, copies data from data loader.
        DNN takes input with format (n, 784).
        '''

        super().load_data()

        self.train_data = self.train_data.astype('float32') / 255
        self.train_data = self.train_data.reshape(self.train_data.shape[0], 28*28)
        self.test_data = self.test_data.astype('float32') / 255
        self.test_data = self.test_data.reshape(self.test_data.shape[0], 28*28)

    def build_model(self):
        '''
        Returns the DNN network to classify FashionMNIST datasets.
        This network can reach 89.51% accuracy on test set of FashionMNIST datasets.
        (the exact accuracy may vary based on intial seed).
        '''

        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=(28*28,)))
        model.add(Dense(512, activation='relu', input_shape=(28*28,)))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def format_data(self, data):
        '''
        Checks and reshapes input data into (n, 784) format before
        feeding it into DNN network.
        '''

        if data.shape[:1] != (28*28,):
            data = data.reshape(data.shape[0], 28*28)
        return data