import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, concatenate
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
import yfinance as yf
import tensorflow as tf
# from pandas_datareader.yahoo.daily import YahooDailyReader
from stockstats import StockDataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from keras import Input
from datetime import datetime, date



# Define the list of stock tickers and the date range
stock_tickers = ['OXY']#, 'GOOGL', 'MSFT']  # Replace with your list of stock tickers
start_date = '2014-01-01'  # Replace with your desired start date
end_date = '2023-09-10'  # Replace with your desired end date
# end_date = datetime.today()

ndx = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
dji = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
dji = dji[1]
ndx = ndx[4]
sp500 = sp500[0]
ndx_tickers = ndx['Ticker']
sp_tickers = sp500['Symbol']
dji_tickers = dji['Symbol']
etf_tickers = pd.Series(['QQQ', 'VOO', 'DIA'])
tickers = pd.concat([ndx_tickers, sp_tickers, dji_tickers, etf_tickers],ignore_index=True).drop_duplicates()
tickers = tickers.sort_values(ascending=True)

# tickers = ndx_tickers

# tickers = ['mrk']



start_date = datetime(2004,1,1)
end_date = date.today()


# Function to preprocess the data for LSTM model
def preprocess_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data['scaled_close'] = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    return stock_data


def preprocess_y(y):
    scaler = MinMaxScaler()
    reshaped_y = y.reshape(-1, 1)
    scaled_y = scaler.fit_transform(reshaped_y)
    # scaled_y = scaled.reshape(1, -1)
    print(scaled_y)
    return scaled_y

# Function to create sequences for training
def create_sequences(data, sequence_length, prediction_window):
    X = []
    y = []
    for i in range(len(data) - sequence_length - prediction_window + 1):
        X.append(data[i:i+sequence_length])
        future_price = data[i+sequence_length:i+sequence_length+prediction_window]
        current_price = data[i+sequence_length-1]
        price_change = future_price - current_price
        y.append(1 if np.any(price_change > 0) else 0)
    return np.array(X), np.array(y)

def create_time_steps(length):
    return list(range(-length, 0))

# def multi_step_plot(history, true_future, prediction):
#     plt.figure(figsize=(18, 6))
#     num_in = create_time_steps(len(history))
#     num_out = len(true_future)

#     plt.plot(num_in, np.array(history[:, 0]), label='History')
#     # if true_future.any():
#     plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
#         label='True Future')
#     if prediction.any():
#         plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
#                  label='Predicted Future')
#     plt.legend(loc='upper left')
#     plt.show()

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

# def plot_train_history(history, title):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']

#     epochs = range(len(loss))

#     plt.figure()

#     plt.plot(epochs, loss, 'b', label='Training loss')
#     plt.plot(epochs, val_loss, 'r', label='Validation loss')
#     plt.title(title)
#     plt.legend()

#     plt.show()

# def multinew_data(dataset, target, start_index, end_index, history_size,
#                       target_size, step, single_step=False):
#     data = []
#     labels = []   

#     # start_index = start_index + history_size
#     start_index = -20
#     if end_index is None:
#         end_index = len(dataset) 

#     for i in range(start_index, end_index-4):
#         indices = range(i-history_size, i, step)
#         data.append(dataset[indices])

#     return np.array(data)

# def prediction_plot(history, prediction):
#     plt.figure(figsize=(18, 6))
#     num_in = create_time_steps(len(history))
#     num_out = len(prediction[0])

#     plt.plot(num_in, np.array(history), label='History')
#     if prediction.any():
#         # print(np.shape(np.arange(num_out)/STEP))
#         # print(np.array(prediction)[0])
#         plt.plot(np.arange(num_out)/STEP, np.array(prediction)[0], 'ro',
#                  label='Predicted Future')
#     plt.legend(loc='upper left')
#     plt.show()





past_history = 30
future_target = 5
STEP = 1



# for ticker in tickers:

# df = YahooDailyReader(ticker, start='2018-01-01', end=datetime.now())
ticker = stock_tickers[0]
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
df = StockDataFrame.retype(df)



features_considered = ['close','atr']
features = df[features_considered]
features.index = df.index
# features.head()

# features.plot(subplots=True,figsize=[12,10])

# Assuming your dataframe is called 'df' with columns 'column1', 'column2', 'column3'
columns_to_normalize = ['close','atr']

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the selected columns
scaler.fit(features[columns_to_normalize])

# Transform the selected columns using the scaler
features[columns_to_normalize] = scaler.transform(features[columns_to_normalize])

dataset = features.values

# Define the function to create the LSTM model
def create_model(learning_rate=0.001, lstm_size=int(32), dropout_rate=0.3, evaluation_step=10):
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(lstm_size,
                                            return_sequences=True,
                                            recurrent_dropout=0.3,
                                            input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.Dropout(0.4))
    multi_step_model.add(Dense(lstm_size/2))

    multi_step_model.add(tf.keras.layers.LSTM(lstm_size/2,recurrent_dropout=0.3))
    multi_step_model.add(tf.keras.layers.Dropout(0.4))
    multi_step_model.add(Dense(lstm_size/4))

    multi_step_model.add(Dense(1,activation='linear'))



    multi_step_model.compile(optimizer=tf.keras.optimizers.legacy.Nadam(learning_rate=0.001),loss='mae', metrics='accuracy')

    return model

param_grid = {
    'batch_size': [128,64,32],
    'learning_rate': [0.0001,0.0005,0.001],
    # 'epochs': [50, 100, 150],
    'evaluation_step': [10,20,30],
    'lstm_size': [32,64,128]
}

# Hyperparameters
batch_size = 128
BUFFER_SIZE = 10000
TRAIN_SPLIT = int(df.shape[0] * 0.8)



# Reproducibility
SEED = 42
tf.random.set_seed(SEED)


x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                TRAIN_SPLIT, past_history,
                                                future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                            TRAIN_SPLIT, None, past_history,
                                            future_target, STEP)

# LSTM Parameters
EPOCHS = 50
PATIENCE = 5

EVALUATION_INTERVAL = len(x_train_multi)//batch_size//2 #15
VALIDATION_INTERVAL = len(x_val_multi)//batch_size//2

# train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
# train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(batch_size).repeat()

# val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
# val_data_multi = val_data_multi.batch(batch_size).repeat()


    # multi_step_model = tf.keras.models.Sequential()
    # multi_step_model.add(tf.keras.layers.LSTM(32,
    #                                         return_sequences=True,
    #                                         recurrent_dropout=0.3,
    #                                         input_shape=x_train_multi.shape[-2:]))
    # multi_step_model.add(tf.keras.layers.Dropout(0.4))
    # multi_step_model.add(Dense(16))

    # multi_step_model.add(tf.keras.layers.LSTM(16,recurrent_dropout=0.3))
    # multi_step_model.add(tf.keras.layers.Dropout(0.4))
    # multi_step_model.add(Dense(8))

    # multi_step_model.add(Dense(1,activation='linear'))



    # multi_step_model.compile(optimizer=tf.keras.optimizers.legacy.Nadam(learning_rate=0.001),loss='mae', metrics='accuracy')
    # print(multi_step_model.summary())


    # # print(model.summary())


    # early_stopping = EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights=True)
    # multi_step_history = model.fit(train_data_multi,
    #                                         epochs=EPOCHS,
    #                                         steps_per_epoch=EVALUATION_INTERVAL,
    #                                         validation_data=val_data_multi,
    #                                         validation_steps=VALIDATION_INTERVAL,
    #                                         callbacks=[early_stopping])

    
    # model.save(f'model/stacked_model_{ticker}.h5')









x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

x_val_multi = x_val_multi[-20:]
y_val_multi = y_val_multi[-20:]

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))

val_data_multi = val_data_multi.batch(batch_size)#.repeat()




# Create the KerasRegressor wrapper 
model = KerasRegressor(build_fn=create_model, lstm_size=[32], learning_rate=[0.0005], evaluation_step=[30])

# Perform randomized search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, refit=True,n_jobs=8,cv=5, random_state=0, error_score='raise', return_train_score=True)  
random_search.fit(x_train_multi, y_train_multi)   
print(np.shape(x_train_multi),np.shape(x_train_multi))

# Print best results
print("Best Hyperparameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)


# #256 0.001 20 0.4
# #512 0.0005 25 0.4
# #512 0.0005 30