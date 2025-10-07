import numpy as np
from data_utils import load_data, create_non_overlapping_windows, download_dji_data
from models import ESN, LSTMModel, TransformerModel
from train_eval import train_and_evaluate

def main(
    data_path=None,
    window_size=10,
    model_type='lstm',
    test_ratio=0.2,
    model_params={},
    fit_params={},
    use_dji_data=False,
    dji_csv_path='dji_2014_2022.csv'
):
    if use_dji_data:
        # Download and save DJI data if not already present
        import os
        if not os.path.exists(dji_csv_path):
            print("Downloading DJI data...")
            download_dji_data(start='2014-01-01', end='2022-12-31', filename=dji_csv_path)
        data_path = dji_csv_path

    data = load_data(data_path)
    X, y = create_non_overlapping_windows(data, window_size)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_type == 'esn':
        model = ESN(input_size=window_size, **model_params)
    elif model_type == 'lstm':
        model = LSTMModel(input_size=window_size, **model_params)
    elif model_type == 'transformer':
        model = TransformerModel(input_size=window_size, **model_params)
    else:
        raise ValueError("Unknown model type")

    mse, preds = train_and_evaluate(model, X_train, y_train, X_test, y_test, fit_params)
    print(f"{model_type.upper()} Test MSE: {mse:.4f}")

if __name__ == "__main__":
    # Example usage: set use_dji_data=True to fetch DJI data automatically
    main(
        data_path='',  # Not used when use_dji_data=True
        window_size=10,
        model_type='lstm',
        model_params={'hidden_size': 64, 'num_layers': 1},
        fit_params={'epochs': 20, 'lr': 1e-3, 'device': 'cpu'},
        use_dji_data=True,  # Set to True to use DJI data from yfinance
        dji_csv_path='dji_2014_2022.csv'
    )
