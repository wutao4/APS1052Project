import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.metrics import mean_squared_error


def plot_training_curves(history):
    train_loss_values = history.history["loss"]  # training loss
    val_loss_values = history.history["val_loss"]  # validation loss
    epochs = range(1, len(train_loss_values) + 1)
    # Plotting training curves
    plt.clf()
    plt.plot(epochs, train_loss_values, label="Train Loss")
    plt.plot(epochs, val_loss_values, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curves")
    plt.show()


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def create_trading_strategy(predictions):
    signal = np.where(predictions > 0, 1, 0)

    return signal


def concatenate_strat_to_test(test_df, trading_signal, seq_len):
    '''
    Concatenates the trading signal to the test_df
    '''
    new_df = test_df.copy()

    # Start and stop length. Start at the seq_len or lookback window - 1
    # This is because if the lookback window is set to 27, we are looking
    # at the last 26 and then predicting for the 27th
    new_signal = np.hstack(([np.nan], trading_signal))
    start = seq_len - 2
    stop = start + len(new_signal)

    # Add the signal to the dataframe
    new_df = new_df.iloc[start:stop, :]
    new_df['signal'] = new_signal.reshape(-1, 1)

    return new_df


def compute_returns(df, price_col):
    '''
    Assumes that the signal is for that day i.e. if a signal of 
    1 exists on the 12th of January, I should buy before that day begins
    '''
    new_df = df.copy()

    new_df['mkt_returns'] = new_df[price_col].pct_change()
    new_df['system_returns'] = new_df['mkt_returns'] * new_df['signal']

    new_df['system_equity'] = np.cumprod(1 + new_df.system_returns) - 1
    new_df['mkt_equity'] = np.cumprod(1 + new_df.mkt_returns) - 1

    return new_df


def plot_returns(df):
    df[['system_equity', 'mkt_equity']].plot()
    plt.show()


def compute_metrics(df):
    new_df = df.copy()

    new_df['system_equity'] = np.cumprod(1 + new_df.system_returns) - 1
    system_cagr = (1 + new_df.system_equity.tail(n=1)) ** (252 / new_df.shape[0]) - 1
    new_df.system_returns = np.log(new_df.system_returns + 1)
    system_sharpe = np.sqrt(252) * np.mean(new_df.system_returns) / np.std(new_df.system_returns)

    new_df['mkt_equity'] = np.cumprod(1 + new_df.mkt_returns) - 1
    mkt_cagr = (1 + new_df.mkt_equity.tail(n=1)) ** (252 / new_df.shape[0]) - 1
    new_df.mkt_returns = np.log(new_df.mkt_returns + 1)
    mkt_sharpe = np.sqrt(252) * np.mean(new_df.mkt_returns) / np.std(new_df.mkt_returns)

    system_metrics = (system_cagr[0], system_sharpe)
    market_metrics = (mkt_cagr[0], mkt_sharpe)

    return system_metrics, market_metrics


def last_time_step_mse(Y_true, Y_pred):
    return mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


# convert history into a cube for series to series prediction
def to_supervised(data, n_input, n_out):
    n_input = n_input + n_out
    X = list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])  # process all columns
        # move along one time step
        in_start += 1
    return np.array(X)
