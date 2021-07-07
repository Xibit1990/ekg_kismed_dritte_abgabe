from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow import split
from tensorflow import squeeze
from tensorflow.keras.regularizers import L2

'''
Load this functions for Lambda layers
'''

def squeeze_layer(x):
    import tensorflow as tf
    return squeeze(x, -1)

def split_att(x):
    import tensorflow as tf
    return split(x, num_or_size_splits=2, axis=2)
