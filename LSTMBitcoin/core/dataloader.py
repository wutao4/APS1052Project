# Tao Wu, Zhoujie (Terrence) Zhao
#
# The data loader class for reading data and creating train/test sets for each model specified in config.json.

import pandas as pd
import numpy as np


class DataLoader:
    """
    Dataloader for creating the train and test sets
    """
    def __init__(self, dataset_fp, train_size, col_name_list, model_type, fut_steps=1):
        self.df = pd.read_csv(dataset_fp, infer_datetime_format=True, parse_dates=['Date'], index_col=['Date'])
        self.index_split = int(train_size*len(self.df))
        self.train_df = self.df[col_name_list].iloc[:self.index_split, :]
        self.test_df = self.df[col_name_list].iloc[self.index_split:, :]
        self.train_data = self.train_df.values
        self.test_data = self.test_df.values
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.fut_steps = fut_steps
        self.model_type = model_type
    
    def get_train_data(self, lookback_window, normalize):
        """
        Get training data for LSTM
        """
        data_x = []
        data_y = []
        for i in range(self.train_len - lookback_window - self.fut_steps):
            x, y = self._next_window(i, lookback_window + self.fut_steps, normalize)
            data_x.append(x)
            data_y.append(y)

        # modify the target data shape for sequence-to-sequence models
        if self.model_type in ['seq2seq', 'gruconv']:
            data_y = np.array(data_y)
            Y = np.empty((len(data_x), lookback_window, self.fut_steps))
            for step_ahead in range(self.fut_steps):
                Y[..., step_ahead] = data_y[..., step_ahead:step_ahead + lookback_window, 0]
            data_y = Y

        # filter the target data for GRU+conv model due to the convolution
        if self.model_type == 'gruconv':
            data_y = data_y[:, 3::2]

        return np.array(data_x), np.array(data_y)
    
    def get_test_data(self, lookback_window, normalize):
        """
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        """
        data_windows = []
        for i in range(self.test_len - lookback_window - self.fut_steps):
            data_windows.append(self.test_data[i:i + lookback_window + self.fut_steps])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows

        x = data_windows[:, :-self.fut_steps]
        y = data_windows[:, -self.fut_steps, 0]

        # modify the target data shape for sequence-to-sequence models
        if self.model_type in ['seq2seq', 'gruconv']:
            y = data_windows[:, lookback_window:, 0]
        
        return x, y
    
    def _next_window(self, i, lookback_window, normalize):
        """
        Generates the next data window
        """
        
        window = self.train_data[i:i+lookback_window]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        x = window[:-self.fut_steps]
        y = window[-self.fut_steps, [0]]

        # modify the target data shape for sequence-to-sequence models
        if self.model_type in ['seq2seq', 'gruconv']:
            y = window[:, [0]]
        
        return x, y
    
    def normalize_windows(self, window_data, single_window=False):
        """
        Normalize window with a base value of zero
        """
        normalized_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalized_window = []
            for col_i in range(window.shape[1]):
                normalized_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalized_window.append(normalized_col)
            # reshape and transpose array back into original multidimensional format
            normalized_window = np.array(normalized_window).T
            normalized_data.append(normalized_window)
        return np.array(normalized_data)
