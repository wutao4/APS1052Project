import os
import time
import logging
import datetime as dt
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, TimeDistributed
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from core.utils import last_time_step_mse


class LSTMTimeSeriesModel:
    """
    Class for building the LSTM
    """
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        """
        Loading the model from a filepath
        """
        logging.info(f"Loading model from {filepath}")
        self.model = load_model(filepath)

    def build_model(self, config):
        """
        Function to build the model from a config file
        """
        logging.info("[MODEL]: Building model...")
        now = time.time()

        model_type = config['model']['type']

        for layer in config['model']['layers'][model_type]:
            units = layer['units'] if 'units' in layer else None
            dropout = layer['dropout'] if 'dropout' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            seq_len = layer['seq_len'] - 1 if 'seq_len' in layer else None
            num_features = layer['num_features'] if 'num_features' in layer else None
            layer_type = layer['type'] if 'type' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None

            if layer_type == 'Dense':
                self.model.add(Dense(units=units, activation=activation))
            elif layer_type == 'TimeDistDense':
                self.model.add(TimeDistributed(Dense(units=config['data']['fut_steps'], activation=activation)))
            elif layer_type == 'LSTM':
                self.model.add(LSTM(units=units,
                                    activation=activation,
                                    input_shape=(seq_len, num_features),
                                    return_sequences=return_seq
                                    ))
            elif layer_type == 'SimpleRNN':
                self.model.add(SimpleRNN(units=units,
                                         activation=activation,
                                         input_shape=(seq_len, num_features),
                                         return_sequences=return_seq
                                         ))
            elif layer_type == 'Dropout':
                self.model.add(Dropout(rate=dropout))

        metrics = [last_time_step_mse] if model_type == 'seq2seq' else None
        self.model.compile(loss=config['model']['loss'], optimizer=config['model']['optimizer'], metrics=metrics)

        time_taken = time.time() - now
        logging.info(f"Model Building complete in {time_taken // 60} min and {(time_taken % 60):.1f} s")

    def train(self, x_train, y_train, config):
        """
        Function to train model
        """
        epochs = config["training"]["epochs"]
        batch_size = config["training"]["batch_size"]
        save_dir = config["model"]["save_dir"]
        model_type = config["model"]["type"]

        save_fname = os.path.join(save_dir, '%s-%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), model_type))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, **config["model"]["checkpoint_params"]),
            ReduceLROnPlateau(**config["model"]["reduce_lr_params"]),
            EarlyStopping(**config["model"]["early_stopping_params"]),
        ]
        logging.info("[MODEL]: Training started")
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=config["training"]["val_split"],
            callbacks=callbacks
        )
        self.model.save(save_fname)

        logging.info(f"Model training completed. Model saved to {save_fname}")

        return history

    def predict_point_by_point(self, data):
        """
        Making one prediction for each sequence
        """
        logging.info('[MODEL]: Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))

        return predicted

    def predict_seq_to_seq(self, data):
        """

        """
        logging.info('[MODEL]: Predicting Sequence-to-Sequence...')
        predicted = self.model.predict(data)
        print(predicted.shape)
        predicted = predicted[:, -1, 0]
        print(predicted.shape)
        predicted = np.reshape(predicted, (predicted.size,))

        return predicted
