# -*- coding: utf-8 -*-

import pickle
from abc import abstractmethod

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical

from base_classifier import BaseClassifier
from graph_utils import draw_history_from_path, set_graph_layout, show_graph


class BaseANNClassifier(BaseClassifier):
    '''
    Base class for all classifier using artificial neural network.
    '''

    def __init__(self, load_data=True):
        '''
        Each neural network model has their own input format,
        this method only call parent's constructor,
        further formatting will be done in concrete class level.
        '''

        super().__init__(load_data)

    @abstractmethod
    def build_model(self):
        '''
        Each concrete class needs to override this method
        and returns its own deep learning network.
        '''

        pass

    def load_data(self):
        '''
        Uses one-hot encoding to encode training and test labels.
        If further formatting of FashionMNIST datasets is needed, 
        please override this method.
        '''

        super().load_data()

        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)

    def train_model(self, epochs, save_path, save_history=False):
        '''
        The first 5000 observations in training set of FashionMNIST
        will be used as validation set, the remaining 45000 observations
        will be used to train the model.
        After 'epochs', Model with highest validation accuracy will be saved to 'save_path'.
        If 'save_history' is set to True, validation loss and accuracy will be saved to disk.
        '''

        if self.train_data is None or self.train_labels is None:
            raise ValueError('Fashion MNIST datasets is not loaded') 

        self.model = self.build_model()

        callbacks_list = [
            ModelCheckpoint(
                filepath=save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]

        x_val = self.train_data[:5000]
        y_val = self.train_labels[:5000]
        x_train = self.train_data[5000:]
        y_train = self.train_labels[5000:]

        history = self.model.fit(x_train, y_train,epochs=epochs, batch_size=128, callbacks=callbacks_list, validation_data=(x_val, y_val))

        if save_history:
            with open('history/{0}_{1}.pkl'.format(
                    self.get_class_name(), self.get_time_stamp()), 'wb') as f:
                f.write(pickle.dumps(history))

    def check_loss_accuracy(self, history_path):
        '''
        Draws graph of validation loss/accuracy vs epochs count.
        Useful when checking overfitting.
        '''

        fig, axes = set_graph_layout(nrows=1, ncols=2, figsize=(12, 6))
        draw_history_from_path(history_path, 'loss', show=False, drw_obj=axes[0])
        draw_history_from_path(history_path, 'accuracy', show=False, drw_obj=axes[1])
        fig.tight_layout()
        show_graph()

    def test_acc(self):
        '''
        Tests pre-trained model using test set of FashionMNIST datasets.
        '''

        if self.test_data is None or self.test_labels is None:
            raise ValueError('Fashion MNIST datasets is not loaded') 

        if not self.model:
            raise ValueError('Please load a model before performing classification.')

        loss, acc = self.model.evaluate(self.test_data, self.test_labels)
        return {'test_loss': loss, 'test_acc':acc}

    def load_model(self, model_path):
        '''
        Loads pre-trained model from storage.
        '''

        self.model = load_model(model_path)

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
        pred = self.model.predict_classes(data)
        return pred
