import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Function to load data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data['Close']
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to build and train LSTM model
def train_lstm_model(train_scaled, seq_length):
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential()
    model.add(Input(shape=(seq_length, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    return model

# Function to predict using LSTM model
def predict_lstm_model(model, data_scaled, scaler, seq_length):
    X, _ = create_sequences(data_scaled, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)
    return pred

# Streamlit app
st.title('Stock Price Prediction Using LSTM')
st.sidebar.header('User Input Parameters')

# User input
tickers = st.sidebar.text_input('Enter stock tickers (comma-separated)', 'AAPL, MSFT, GOOGL')
start_date = st.sidebar.date_input('Start date', pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input('End date', pd.to_datetime('2023-01-01'))

tickers = [ticker.strip() for ticker in tickers.split(',')]
seq_length = 50

# Button to trigger prediction
if st.button('Predict Stock Prices'):
    for ticker in tickers:
        st.subheader(f'{ticker} Stock Price Prediction')

        # Load and prepare data
        data = load_data(ticker, start_date, end_date)
        if data is None:
            continue

        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(np.array(train).reshape(-1, 1))
        test_scaled = scaler.transform(np.array(test).reshape(-1, 1))

        # Train LSTM model
        model = train_lstm_model(train_scaled, seq_length)

        # Predict
        train_pred = predict_lstm_model(model, train_scaled, scaler, seq_length)
        test_pred = predict_lstm_model(model, test_scaled, scaler, seq_length)

        # Calculate RMSE
        train_rmse = np.sqrt(mean_squared_error(train[seq_length:], train_pred))
        test_rmse = np.sqrt(mean_squared_error(test[seq_length:], test_pred))
        st.write(f'Train RMSE: {train_rmse:.4f}')
        st.write(f'Test RMSE: {test_rmse:.4f}')

        # Display predicted values
        st.write("Predicted values for test data:")
        st.write(pd.DataFrame(test_pred, index=data.index[train_size + seq_length:], columns=["Predicted Price"]))

        # Plot predictions
        plt.figure(figsize=(14, 7))
        plt.plot(data.index[seq_length:train_size], train[seq_length:], label='Train True')
        plt.plot(data.index[seq_length:train_size], train_pred, label='Train Predict')
        plt.plot(data.index[train_size + seq_length:], test[seq_length:], label='Test True')
        plt.plot(data.index[train_size + seq_length:], test_pred, label='Test Predict')
        plt.legend()
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.grid()

        st.pyplot(plt)
        plt.close()
