from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

app = Flask(__name__)

# Define global variables
time_step = 5  # You can adjust this according to your needs
model = None

def create_and_train_model(train_data, test_data):
    global model
    # Create and compile the model
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(32, return_sequences=True))
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    model.fit(train_data[0], train_data[1], validation_data=(test_data[0], test_data[1]), epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        company = request.form['company'].upper()
        stock_exchange = request.form['stock_exchange'].upper()

        # Initialize TvDatafeed
        tv = TvDatafeed()

        # Get historical data for the company
        history = tv.get_hist(symbol=company, exchange=stock_exchange, interval=Interval.in_daily, n_bars=4000)

        # Convert historical data to DataFrame
        df = pd.DataFrame(history)

        # Preprocessing
        print("Before: ", df.head())
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        # Print DataFrame after converting index
        print("After converting index to datetime: ", df.head())
        # Sort DataFrame by date
        df.sort_index(inplace=True)
        closedf = df['close']
        scaler = MinMaxScaler(feature_range=(0, 1))
        closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

        # Split data into train and test sets
        training_size = int(len(closedf) * 0.7)
        test_size = len(closedf) - training_size
        train_data, test_data = closedf[0:training_size], closedf[training_size:len(closedf)]


        # convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)
        
        # Reshape data for LSTM
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Train the model
        create_and_train_model((X_train, y_train),(X_test, y_test))

        # Predict next 10 days
        pred_days = 10
        lst_output = []
        x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        for i in range(pred_days):
            if len(temp_input) > time_step:
                x_input=np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, time_step, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
            else:
                x_input = x_input.reshape((1, time_step, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())

        # Inverse transform to get actual prices
        lst_output = scaler.inverse_transform(lst_output)

        # Display predicted prices
        predictions = [round(float(price), 2) for price in lst_output.flatten()]

        return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
