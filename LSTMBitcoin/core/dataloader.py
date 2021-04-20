import pandas as pd
import numpy as np


class DataLoader:
    '''
    Dataloader for creating the train and test sets
    '''
    def __init__(self, dataset_fp, train_size, col_name_list, fut_steps=0):
        self.df = pd.read_csv(dataset_fp, infer_datetime_format=True, parse_dates=['Date'], index_col=['Date'])
        self.index_split = int(train_size*len(self.df))
        self.train_df = self.df[col_name_list].iloc[:self.index_split,:]
        self.test_df = self.df[col_name_list].iloc[self.index_split:,:]
        self.train_data = self.train_df.values
        self.test_data = self.test_df.values
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        self.fut_steps = fut_steps
    
    def get_train_data(self, lookback_window, normalize):
        '''
        Get training data for LSTM
        '''
        data_x = []
        data_y = []
        lookback_window += self.fut_steps
        for i in range(self.train_len - lookback_window):
            x, y = self._next_window(i, lookback_window, normalize)
            data_x.append(x)
            data_y.append(y)

        if self.fut_steps != 0:
            # data_y = np.array(data_y).squeeze()
            pass
        return np.array(data_x), np.array(data_y)
    
    def get_test_data(self, lookback_window, normalize):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.test_len - lookback_window):
            data_windows.append(self.test_data[i:i+lookback_window])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalize else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        
        return x, y
    
    def _next_window(self, i, lookback_window, normalize):
        '''
        Generates the next data window
        '''
        
        window = self.train_data[i:i+lookback_window]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        # print(window.shape)
        x = window[:-1 - self.fut_steps]
        y = window[-1 - self.fut_steps:-1, [0]]
        
        return x, y
    
    def normalize_windows(self, window_data, single_window=False):
        '''
        Normalize window with a base value of zero
        '''
        normalized_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalized_window = []
            for col_i in range(window.shape[1]):
                normalized_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalized_window.append(normalized_col)
            normalized_window = np.array(normalized_window).T # reshape and transpose array back into original multidimensional format
            normalized_data.append(normalized_window)
        return np.array(normalized_data)