# -*- coding: utf-8 -*-

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from base_ann_classifier import BaseANNClassifier


class CNNClassifier(BaseANNClassifier):
    '''
    Image classification using convolutional neural network (CNN).
    Can reach 92.01% accuracy on test set of FashionMNIST datasets.

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
        CNN takes input with format (n, 28, 28, 1).
        '''

        super().load_data()

        self.train_data = self.train_data.astype('float32') / 255
        self.train_data = self.train_data.reshape(self.train_data.shape[0], 28, 28, 1)
        self.test_data = self.test_data.astype('float32') / 255
        self.test_data = self.test_data.reshape(self.test_data.shape[0], 28, 28, 1)

    def build_model(self):
        '''
        Returns the CNN network to classify FashionMNIST datasets.
        This network can reach 92.01% accuracy on test set of FashionMNIST datasets.
        (the exact accuracy may vary based on intial seed).
        '''

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def format_data(self, data):
        '''
        Checks and reshapes input data into (n, 28, 28, 1) format before
        feeding it into CNN network.
        '''

        if data.shape[:1] != (28, 28, 1):
            data = data.reshape(data.shape[0], 28, 28, 1)
        return data
