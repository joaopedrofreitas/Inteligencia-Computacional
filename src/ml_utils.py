import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Normalization

def create_mlp(input_shape):
    norm = Normalization(input_shape=input_shape, axis=-1)
    model = Sequential([
        norm,
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))
    return model, norm 