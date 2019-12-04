# -*- coding: utf-8 -*-

'''
Miscellaneous methods to draw graphs and images.
'''

import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._subplots import SubplotBase

def set_graph_layout(nrows, ncols, figsize):
    '''
    Set the size and layout of graph.
    '''

    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

def show_graph():
    '''
    Displays graph using matplotlib.
    '''

    plt.show()

def draw_history_from_path(history_path, attr, show=True, drw_obj=plt, disp_obj=plt):
    '''
    Reads history file of keras models from storage and
    draws validation loss/accuracy vs epochs count graph.
    '''

    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    draw_history(history, attr, show, drw_obj, disp_obj)

def draw_history(history, attr, show=True, drw_obj=plt, disp_obj=plt):
    '''
    Draws validation loss/accuracy vs epochs count graph,
    base on history file of keras models.
    '''

    history_dict = history.history
    values = history_dict['{0}'.format(attr)]
    val_values = history_dict['val_{0}'.format(attr)]

    epochs = range(1, len(history_dict['accuracy']) + 1)

    drw_obj.plot(epochs, values, 'bo', label='Training {0}'.format(attr))
    drw_obj.plot(epochs, val_values, 'b', label='Validation {0}'.format(attr))
    drw_obj.legend()

    if isinstance(drw_obj, SubplotBase):
        drw_obj.set_title('Training and validation {0}'.format(attr))
        drw_obj.set_xlabel('Epochs')
        drw_obj.set_ylabel(attr.title())
    else:
        drw_obj.title('Training and validation {0}'.format(attr))
        drw_obj.xlabel('Epochs')
        drw_obj.ylabel(attr.title())

    if show:
        disp_obj.show()

def display_image(data, height, width):
    '''
    Reads image data from numpy array and displays in photo viewer apps.
    '''

    image = np.array(data, dtype='float')
    pixels = image.reshape((height, width))
    plt.imshow(pixels, cmap='gray')
    plt.show()
