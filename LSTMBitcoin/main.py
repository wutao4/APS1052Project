import os
import json
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # USE THIS TO DISABLE ALL LOGGING INFORMATION FROM TENSORFLOW

from core.utils import *
from core.dataloader import DataLoader
from core.model import LSTMTimeSeriesModel


def main():
    """
    Main app function
    """
    # Trying to set up the app, catching any errors
    # Setting up logging, arg parsing, config file
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Logging set up")

    try:
        logging.info(f"Trying to load config.json...")
        with open("config.json", "r") as f:
            configs = json.load(f)
        logging.info(f"Successfully loaded configuration file!")
    except:
        print("Something went wrong reading the config file. Exiting.")
        return

    # Instantiating the dataloader in the core/dataloader module
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
        model_type=configs['model']['type'],
        fut_steps=configs['data']['fut_steps'] if configs['model']['type'] in ['seq2seq', 'gruconv'] else 1
    )

    # Train x and y
    x, y = data.get_train_data(
        lookback_window=configs['data']['sequence_length'] - 1,
        normalize=configs['data']['normalize']
    )

    # Instantiating the LSTMTimeSeriesModel model defined in the core/model module
    model = LSTMTimeSeriesModel()
    model.build_model(configs)

    # Training the model using the inbuilt method of the LSTMTimeSeriesModel class
    history = model.train(x, y, configs)

    # Plotting the train/val loss curves
    plot_training_curves(history)

    # Extracting the x and y test data
    x_test, y_test = data.get_test_data(
        lookback_window=configs['data']['sequence_length'] - 1,
        normalize=configs['data']['normalize']
    )

    # Predicting the results point by point
    if configs['model']['type'] in ['seq2seq', 'gruconv']:
        predictions = model.predict_seq_to_seq(x_test)
        y_test = y_test[:, 0]
    else:
        x_test = x_test[:-configs['data']['fut_steps'] + 1, :, :]
        y_test = y_test[:-configs['data']['fut_steps'] + 1]
        predictions = model.predict_point_by_point(x_test)

    # Plotting the predictions compared to the actual values
    plot_results(predictions, y_test)

    # Using normalized predictions on test set, create a trading strategy
    signal = create_trading_strategy(predictions)

    # Combine this trading information with the test df. This is not the same as y_test
    # because y_test is smaller in size because of the lookback_window
    new_df = concatenate_strat_to_test(data.test_df, signal, configs["data"]["sequence_length"])

    # Compute the returns using this concatenated df
    new_df = compute_returns(new_df, configs["data"]["price_column"])

    # Compute Sharpe Ratio and CAGR for system and market, market being holding BTC
    system_metrics, market_metrics = compute_metrics(new_df)
    print(f"System CAGR: {system_metrics[0] * 100:.1f}%")
    print(f"System Sharpe: {system_metrics[1]:.1f}")

    print(f"Market CAGR: {market_metrics[0] * 100:.1f}%")
    print(f"Market Sharpe: {market_metrics[1]:.1f}")

    # Plot the results
    plot_returns(new_df)


# This script gets executed
if __name__ == '__main__':
    main()
