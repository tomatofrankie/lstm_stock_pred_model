import yfinance as yf
from datetime import datetime, date, timedelta
import pandas as pd
import logging
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
from stockstats import StockDataFrame
# import modify_excel
# import zip_log
# import upload_gsheet



# Variables
fast_k = 13
fast_smooth = 3
slow_k = 34
slow_smooth = 5

s_overbought = 89
overbought = 80
midline = 50
oversold = 20
s_oversold = 13



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

def multinew_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []   

    # start_index = start_index + history_size
    start_index = -history_size
    if end_index is None:
        end_index = len(dataset) 

    for i in range(start_index, end_index-target_size):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

    return np.array(data)


# Options
# to_excel = input("Update Excel: ")
# while True:
    # if to_excel == "y":
    #     break
    # elif to_excel == "n":
    #     break
    # elif to_excel == "debug":
    #     break
    # else:
    #     to_excel = input("Update Excel: ")


time = datetime.now()
result_df = pd.DataFrame()

# ndx = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
# sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# dji = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
# dji = dji[1]
# ndx = ndx[4]
# sp500 = sp500[0]
# ndx_tickers = ndx['Ticker']
# sp_tickers = sp500['Symbol']
# dji_tickers = dji['Symbol']
# etf_tickers = pd.Series(['QQQ', 'VOO', 'DIA'])
# tickers = pd.concat([ndx_tickers, sp_tickers, dji_tickers, etf_tickers],ignore_index=True).drop_duplicates()
# tickers = tickers.sort_values(ascending=True)
tickers = ['OXY']#, 'GOOGL', 'MSFT']

# tickers = ndx_tickers

# tickers = ['mrk']



start_date = datetime(2020,1,1)
end_date = date.today() # - timedelta(days = 3)
# end_date = datetime(2023,8,1)


# ZIP LOG EVERY MONTH
# zip_log.zip()


# Logging
# logpath = "log/" + time.strftime("%m_%d_%Y") + ".log"
# if to_excel == "debug":
#     logpath = "log/" + time.strftime("%m_%d_%Y") + "debug.log"
#     logging.basicConfig(filename=logpath, encoding='utf-8', level=logging.DEBUG)
# if os.path.exists(logpath):
#     os.remove(logpath)
# if to_excel != "debug":
#     logging.basicConfig(filename=logpath, encoding='utf-8')

# for ticker in tickers:
#     try:
#         net_profit = 0
#         num_of_orders = 0
#         profit_orders = 0
#         order_price = 0
#         total_profit = 0
#         total_loss = 0
#         buy,sell,close_buy,close_sell = False, False, False, False
#         today_buy,today_sell,today_close_buy,today_close_sell = False, False, False, False

#         data = yf.download(ticker, period = '1d', start = start_date, end = end_date, auto_adjust = False)

#         h_fast = pd.Series(data['High']).rolling(window=fast_k).max()
#         l_fast = pd.Series(data['Low']).rolling(window=fast_k).min()
#         RSV_fast = 100 * ((data['Close'] - l_fast) / (h_fast - l_fast))

#         h_slow = pd.Series(data['High']).rolling(window=slow_k).max()
#         l_slow = pd.Series(data['Low']).rolling(window=slow_k).min()
#         RSV_slow = 100 * ((data['Close'] - l_slow) / (h_slow - l_slow))

#         k_fast = RSV_fast.ewm(com=fast_smooth-1, adjust=False).mean()
#         k_slow = RSV_slow.ewm(com=slow_smooth-1, adjust=False).mean()

#         fast_kdj = k_fast.dropna()
#         slow_kdj = k_slow.dropna()

#         # Past performance
#         for i in range(0,len(slow_kdj)-1):


            

#             features_considered = ['Close','High','Low']
#             features = data[features_considered]
#             features.index = data.index


#             # Assuming your dataframe is called 'df' with columns 'column1', 'column2', 'column3'
#             columns_to_normalize = ['Close', 'High', 'Low']

#             # Create an instance of MinMaxScaler
#             scaler = MinMaxScaler()

#             # Fit the scaler on the selected columns
#             scaler.fit(features[columns_to_normalize])

#             # Transform the selected columns using the scaler
#             features[columns_to_normalize] = scaler.transform(features[columns_to_normalize])

#             dataset = features.values

#             past_history = 20
#             future_target = 5
#             STEP = 1


#             x_new_multi = multinew_data(dataset, dataset[:, 2],
#                                                         int(data.shape[0] * 0.8), None, past_history,
#                                                         future_target, STEP)

#             x_new_multi = x_new_multi[-20:]


#             model = load_model(f'model/model_{ticker}.h5')
#             prediction = model.predict(x_new_multi[-1:], verbose=0)
#             last_price_scaled = x_new_multi[-1][-1:,2]


#             profit,loss = 0,0
#             if slow_kdj.iloc[i] > oversold and slow_kdj.iloc[i - 1] < oversold:
#                 if (fast_kdj.iloc[i + slow_k - fast_k - 1] > oversold and fast_kdj.iloc[i + slow_k - fast_k - 2] < oversold) or \
#                 (fast_kdj.iloc[i + slow_k - fast_k - 2] > oversold and fast_kdj.iloc[i + slow_k - fast_k - 3] < oversold) or \
#                 (fast_kdj.iloc[i + slow_k - fast_k - 3] > oversold and fast_kdj.iloc[i + slow_k - fast_k - 4] < oversold) or \
#                 prediction > last_price_scaled:
#                 # (fast_kdj.iloc[i + slow_k - fast_k - 4] > oversold and fast_kdj.iloc[i + slow_k - fast_k - 5] < oversold):
#                     buy = True
#                     order_price = data.iloc[i+slow_k-1]['Close']
#                     num_of_orders += 1
#                     size = round(1000/order_price)
#                     print(slow_kdj.index[i], round(order_price,3), "BUY", size, "lot", ticker)
#                     logging.info("%s %d BUY %i lot %s", slow_kdj.index[i], round(order_price,3), size, ticker)

#             if slow_kdj.iloc[i] < overbought and slow_kdj.iloc[i - 1] > overbought:
#                 if (fast_kdj.iloc[i + slow_k - fast_k - 1] < overbought and fast_kdj.iloc[i + slow_k - fast_k - 2] > overbought) or \
#                 (fast_kdj.iloc[i + slow_k - fast_k - 2] < overbought and fast_kdj.iloc[i + slow_k - fast_k - 3] > overbought) or \
#                 (fast_kdj.iloc[i + slow_k - fast_k - 3] < overbought and fast_kdj.iloc[i + slow_k - fast_k - 4] > overbought) or \
#                 prediction < last_price_scaled:
#                 # (fast_kdj.iloc[i + slow_k - fast_k - 4] < overbought and fast_kdj.iloc[i + slow_k - fast_k - 5] > overbought): 
#                     sell = True
#                     order_price = data.iloc[i+slow_k-1]['Close']
#                     num_of_orders += 1
#                     size = round(1000/order_price)
#                     print(slow_kdj.index[i], round(order_price,3), "SELL", size, "lot", ticker)
#                     logging.info("%s %d SELL %i lot %s", slow_kdj.index[i], round(order_price,3), size, ticker)

#             if buy == True and slow_kdj.iloc[i] < slow_kdj.iloc[i-1]:
#                 close_buy = True
#                 close_price = data.iloc[i+slow_k-1]['Close']
#                 profit = (close_price - order_price)*size
#                 buy = False
#                 print(slow_kdj.index[i], round(data.iloc[i+slow_k-1]['Close'],3), "CLOSE", ticker)
#                 logging.info("%s %d CLOSE %s", slow_kdj.index[i], round(data.iloc[i+slow_k-1]['Close'],3), ticker)

#             if sell == True and slow_kdj.iloc[i] > slow_kdj.iloc[i-1]:
#                 close_sell = True
#                 close_price = data.iloc[i+slow_k-1]['Close']
#                 profit = -(close_price - order_price)*size
#                 sell = False
#                 print(slow_kdj.index[i], round(data.iloc[i+slow_k-1]['Close'],3), "CLOSE", ticker)
#                 logging.info("%s %d CLOSE %s", slow_kdj.index[i], round(data.iloc[i+slow_k-1]['Close'],3), ticker)

#             if profit < 0:
#                 loss = profit
#                 profit = 0

#             elif profit > 0:
#                 profit_orders += 1
            
#             net_profit += profit + loss
#             total_profit += profit
#             total_loss += loss

#         total_loss = -total_loss
#         # print(profit_orders, num_of_orders, net_profit, total_profit, total_loss)

        
#         # Check Signal of today
#         if slow_kdj.iloc[-1] > oversold and slow_kdj.iloc[-2] < oversold:
#             if (fast_kdj.iloc[-1] > oversold and fast_kdj.iloc[-2] < oversold) or \
#             (fast_kdj.iloc[-2] > oversold and fast_kdj.iloc[-3] < oversold) or \
#             (fast_kdj.iloc[-3] > oversold and fast_kdj.iloc[-4] < oversold): # or \
#             # (fast_kdj.iloc[-4] > oversold and fast_kdj.iloc[-5] < oversold):
#                 today_buy = True
#                 close = data.iloc[-1]['Close']
#                 print(round(close,3), "BUY", ticker)
#         if slow_kdj.iloc[-1] < overbought and slow_kdj.iloc[-2] > overbought:
#             if (fast_kdj.iloc[-1] < overbought and fast_kdj.iloc[-2] > overbought) or \
#             (fast_kdj.iloc[-2] < overbought and fast_kdj.iloc[-3] > overbought) or \
#             (fast_kdj.iloc[-3] < overbought and fast_kdj.iloc[-4] > overbought): # or \
#             # (fast_kdj.iloc[-4] < overbought and fast_kdj.iloc[-5] > overbought): 
#                 today_sell = True
#                 close = data.iloc[-1]['Close']
#                 print(round(close,3), "SELL", ticker)
#         if buy == True and slow_kdj.iloc[-1] < slow_kdj.iloc[-2]:
#             today_close_buy = True
#             close = data.iloc[-1]['Close']
#             print(round(close,3), "CLOSE", ticker)
#         if sell == True and slow_kdj.iloc[-1] > slow_kdj.iloc[-2]:
#             today_close_sell = True
#             close = data.iloc[-1]['Close']
#             print(round(close,3), "CLOSE", ticker)
#     except:
#         pass
#     finally:
#         belongs = ''
#         # if ticker in ndx_tickers.values:
#         #     belongs += '.NDX'
#         # if ticker in sp_tickers.values:
#         #     belongs += '.SPX'
#         # if ticker in dji_tickers.values:
#         #     belongs += '.DJI'
#         # if ticker in etf_tickers.values:
#         #     belongs += 'ETF'
        
#         print('Number of Trades', round(num_of_orders), 'Profit', round(net_profit,3), 'Profit %', round(profit_orders/num_of_orders*100,3), 
#                 'Profit Factor', round(total_profit/total_loss,3))
#         try:
#             # result_df = result_df.append({'Stock': ticker, 'Belongs': belongs, 'Number of Trades': round(num_of_orders), 'Profit': round(net_profit,3), 
#             #                 'Profit %': round(profit_orders/num_of_orders*100,3), 'Profit Factor': round(total_profit/total_loss,3), 
#             #                 'BUY': today_buy, 'SELL': today_sell, 'CLOSE BUY': today_close_buy, 'CLOSE SELL': today_close_sell},ignore_index=True)
#             # if (today_buy == True or today_sell == True or today_close_buy == True or today_close_sell == True) and net_profit > 0 and data.iloc[-1]['Close'] > 20:
#                 # print(ticker)
#                 # result_df = result_df.append({'Stock': ticker, 'Belongs': belongs, 'Number of Trades': round(num_of_orders), 'Profit': round(net_profit,3), 
#                 #             'Profit %': round(profit_orders/num_of_orders*100,3), 'Profit Factor': round(total_profit/total_loss,3), 
#                 #             'BUY': today_buy, 'SELL': today_sell, 'CLOSE BUY': today_close_buy, 'CLOSE SELL': today_close_sell, 
#                 #             'Last Close': round(close,3)},ignore_index=True)
#                 # result_df = pd.concat([result_df, pd.DataFrame([{'Stock': ticker, 'Belongs': belongs, 'Number of Trades': round(num_of_orders),
#                 #                                                 'Profit': round(net_profit,3), 'Profit %': round(profit_orders/num_of_orders*100,3), 
#                 #                                                 'Profit Factor': round(total_profit/total_loss,3), 'BUY': today_buy, 'SELL': today_sell, 
#                 #                                               'CLOSE BUY': today_close_buy, 'CLOSE SELL': today_close_sell, 'Last Close': round(close,3)}])],ignore_index=True)
#             result_df = pd.concat([result_df, pd.DataFrame([{'Stock': ticker, 'Number of Trades': round(num_of_orders),
#                                                                 'Profit': round(net_profit,3), 'Profit %': round(profit_orders/num_of_orders*100,3), 
#                                                                 'Profit Factor': round(total_profit/total_loss,3), 'BUY': today_buy, 'SELL': today_sell, 
#                                                               'CLOSE BUY': today_close_buy, 'CLOSE SELL': today_close_sell, 'Last Close': round(close,3)}])],ignore_index=True)

#         except:
#             pass


for ticker in tickers:
    net_profit = 0
    num_of_orders = 0
    profit_orders = 0
    order_price = 0
    total_profit = 0
    total_loss = 0
    buy,sell,close_buy,close_sell = False, False, False, False
    today_buy,today_sell,today_close_buy,today_close_sell = False, False, False, False

    data = yf.download(ticker, period = '1d', start = start_date, end = end_date, auto_adjust = False)

    h_fast = pd.Series(data['High']).rolling(window=fast_k).max()
    l_fast = pd.Series(data['Low']).rolling(window=fast_k).min()
    RSV_fast = 100 * ((data['Close'] - l_fast) / (h_fast - l_fast))

    h_slow = pd.Series(data['High']).rolling(window=slow_k).max()
    l_slow = pd.Series(data['Low']).rolling(window=slow_k).min()
    RSV_slow = 100 * ((data['Close'] - l_slow) / (h_slow - l_slow))

    k_fast = RSV_fast.ewm(com=fast_smooth-1, adjust=False).mean()
    k_slow = RSV_slow.ewm(com=slow_smooth-1, adjust=False).mean()

    fast_kdj = k_fast.dropna()
    slow_kdj = k_slow.dropna()

    # Past performance
    for i in range(0,len(slow_kdj)-1):


        data = StockDataFrame.retype(data)

        features_considered = ['close']
        features = data[features_considered]
        features.index = data.index


        # Assuming your dataframe is called 'df' with columns 'column1', 'column2', 'column3'
        columns_to_normalize = ['close']

        # Create an instance of MinMaxScaler
        scaler = MinMaxScaler()

        # Fit the scaler on the selected columns
        scaler.fit(features[columns_to_normalize])

        # Transform the selected columns using the scaler
        features[columns_to_normalize] = scaler.transform(features[columns_to_normalize])

        dataset = features.values

        past_history = 20
        future_target = 5
        STEP = 1


        x_new_multi = multinew_data(dataset, dataset[:, 0],
                                                    int(data.shape[0] * 0.8), None, past_history,
                                                    future_target, STEP)

        x_new_multi = x_new_multi[-20:]


        model = load_model(f'model/model_OXY.h5')
        prediction = model.predict(x_new_multi[-1:], verbose=0)
        prediction_0 = model.predict(x_new_multi[-2:-1], verbose=0)
        #last_price_scaled = x_new_multi[-1][-1:,0]


        profit,loss = 0,0
        if slow_kdj.iloc[i] > oversold and slow_kdj.iloc[i - 1] < oversold and prediction > prediction_0:
            if (fast_kdj.iloc[i + slow_k - fast_k - 1] > oversold and fast_kdj.iloc[i + slow_k - fast_k - 2] < oversold) or \
            (fast_kdj.iloc[i + slow_k - fast_k - 2] > oversold and fast_kdj.iloc[i + slow_k - fast_k - 3] < oversold) or \
            (fast_kdj.iloc[i + slow_k - fast_k - 3] > oversold and fast_kdj.iloc[i + slow_k - fast_k - 4] < oversold):# or \
            # prediction > last_price_scaled:
            # (fast_kdj.iloc[i + slow_k - fast_k - 4] > oversold and fast_kdj.iloc[i + slow_k - fast_k - 5] < oversold):
                buy = True
                order_price = data.iloc[i+slow_k-1]['close']
                num_of_orders += 1
                size = round(1000/order_price)
                print(slow_kdj.index[i], round(order_price,3), "BUY", size, "lot", ticker)
                logging.info("%s %d BUY %i lot %s", slow_kdj.index[i], round(order_price,3), size, ticker)

        if slow_kdj.iloc[i] < overbought and slow_kdj.iloc[i - 1] > overbought and prediction < prediction_0:
            if (fast_kdj.iloc[i + slow_k - fast_k - 1] < overbought and fast_kdj.iloc[i + slow_k - fast_k - 2] > overbought) or \
            (fast_kdj.iloc[i + slow_k - fast_k - 2] < overbought and fast_kdj.iloc[i + slow_k - fast_k - 3] > overbought) or \
            (fast_kdj.iloc[i + slow_k - fast_k - 3] < overbought and fast_kdj.iloc[i + slow_k - fast_k - 4] > overbought):# or \
            # prediction < last_price_scaled:
            # (fast_kdj.iloc[i + slow_k - fast_k - 4] < overbought and fast_kdj.iloc[i + slow_k - fast_k - 5] > overbought): 
                sell = True
                order_price = data.iloc[i+slow_k-1]['close']
                num_of_orders += 1
                size = round(1000/order_price)
                print(slow_kdj.index[i], round(order_price,3), "SELL", size, "lot", ticker)
                logging.info("%s %d SELL %i lot %s", slow_kdj.index[i], round(order_price,3), size, ticker)

        if buy == True and slow_kdj.iloc[i] < slow_kdj.iloc[i-1]:
            close_buy = True
            close_price = data.iloc[i+slow_k-1]['close']
            profit = (close_price - order_price)*size
            buy = False
            print(slow_kdj.index[i], round(data.iloc[i+slow_k-1]['close'],3), "CLOSE", ticker)
            logging.info("%s %d CLOSE %s", slow_kdj.index[i], round(data.iloc[i+slow_k-1]['close'],3), ticker)

        if sell == True and slow_kdj.iloc[i] > slow_kdj.iloc[i-1]:
            close_sell = True
            close_price = data.iloc[i+slow_k-1]['close']
            profit = -(close_price - order_price)*size
            sell = False
            print(slow_kdj.index[i], round(data.iloc[i+slow_k-1]['close'],3), "CLOSE", ticker)
            logging.info("%s %d CLOSE %s", slow_kdj.index[i], round(data.iloc[i+slow_k-1]['close'],3), ticker)

        if profit < 0:
            loss = profit
            profit = 0

        elif profit > 0:
            profit_orders += 1
        
        net_profit += profit + loss
        total_profit += profit
        total_loss += loss

    total_loss = -total_loss
    # print(profit_orders, num_of_orders, net_profit, total_profit, total_loss)

    
    # Check Signal of today
    if slow_kdj.iloc[-1] > oversold and slow_kdj.iloc[-2] < oversold:
        if (fast_kdj.iloc[-1] > oversold and fast_kdj.iloc[-2] < oversold) or \
        (fast_kdj.iloc[-2] > oversold and fast_kdj.iloc[-3] < oversold) or \
        (fast_kdj.iloc[-3] > oversold and fast_kdj.iloc[-4] < oversold): # or \
        # (fast_kdj.iloc[-4] > oversold and fast_kdj.iloc[-5] < oversold):
            today_buy = True
            close = data.iloc[-1]['close']
            print(round(close,3), "BUY", ticker)
    if slow_kdj.iloc[-1] < overbought and slow_kdj.iloc[-2] > overbought:
        if (fast_kdj.iloc[-1] < overbought and fast_kdj.iloc[-2] > overbought) or \
        (fast_kdj.iloc[-2] < overbought and fast_kdj.iloc[-3] > overbought) or \
        (fast_kdj.iloc[-3] < overbought and fast_kdj.iloc[-4] > overbought): # or \
        # (fast_kdj.iloc[-4] < overbought and fast_kdj.iloc[-5] > overbought): 
            today_sell = True
            close = data.iloc[-1]['close']
            print(round(close,3), "SELL", ticker)
    if buy == True and slow_kdj.iloc[-1] < slow_kdj.iloc[-2]:
        today_close_buy = True
        close = data.iloc[-1]['close']
        print(round(close,3), "CLOSE", ticker)
    if sell == True and slow_kdj.iloc[-1] > slow_kdj.iloc[-2]:
        today_close_sell = True
        close = data.iloc[-1]['close']
        print(round(close,3), "CLOSE", ticker)
    
    belongs = ''
    # if ticker in ndx_tickers.values:
    #     belongs += '.NDX'
    # if ticker in sp_tickers.values:
    #     belongs += '.SPX'
    # if ticker in dji_tickers.values:
    #     belongs += '.DJI'
    # if ticker in etf_tickers.values:
    #     belongs += 'ETF'
    
    print('Number of Trades', round(num_of_orders), 'Profit', round(net_profit,3), 'Profit %', round(profit_orders/num_of_orders*100,3), 
            'Profit Factor', round(total_profit/total_loss,3))
    try:
        # result_df = result_df.append({'Stock': ticker, 'Belongs': belongs, 'Number of Trades': round(num_of_orders), 'Profit': round(net_profit,3), 
        #                 'Profit %': round(profit_orders/num_of_orders*100,3), 'Profit Factor': round(total_profit/total_loss,3), 
        #                 'BUY': today_buy, 'SELL': today_sell, 'CLOSE BUY': today_close_buy, 'CLOSE SELL': today_close_sell},ignore_index=True)
        # if (today_buy == True or today_sell == True or today_close_buy == True or today_close_sell == True) and net_profit > 0 and data.iloc[-1]['Close'] > 20:
            # print(ticker)
            # result_df = result_df.append({'Stock': ticker, 'Belongs': belongs, 'Number of Trades': round(num_of_orders), 'Profit': round(net_profit,3), 
            #             'Profit %': round(profit_orders/num_of_orders*100,3), 'Profit Factor': round(total_profit/total_loss,3), 
            #             'BUY': today_buy, 'SELL': today_sell, 'CLOSE BUY': today_close_buy, 'CLOSE SELL': today_close_sell, 
            #             'Last Close': round(close,3)},ignore_index=True)
            # result_df = pd.concat([result_df, pd.DataFrame([{'Stock': ticker, 'Belongs': belongs, 'Number of Trades': round(num_of_orders),
            #                                                 'Profit': round(net_profit,3), 'Profit %': round(profit_orders/num_of_orders*100,3), 
            #                                                 'Profit Factor': round(total_profit/total_loss,3), 'BUY': today_buy, 'SELL': today_sell, 
            #                                               'CLOSE BUY': today_close_buy, 'CLOSE SELL': today_close_sell, 'Last Close': round(close,3)}])],ignore_index=True)
        result_df = pd.concat([result_df, pd.DataFrame([{'Stock': ticker, 'Number of Trades': round(num_of_orders),
                                                            'Profit': round(net_profit,3), 'Profit %': round(profit_orders/num_of_orders*100,3), 
                                                            'Profit Factor': round(total_profit/total_loss,3), 'BUY': today_buy, 'SELL': today_sell, 
                                                            'CLOSE BUY': today_close_buy, 'CLOSE SELL': today_close_sell, 'Last Close': round(close,3)}])],ignore_index=True)

    except:
        pass






result_df = result_df.replace({True: 'True', False: 'False'})
print(result_df)


# if os.path.exists('signal/' + time.strftime("%b_%Y") + '/') != True:
#     os.mkdir('signal/' + time.strftime("%b_%Y") + '/')
# result_df.to_csv(path_or_buf='signal/' + time.strftime("%b_%Y") + '/' + time.strftime("%m_%d_%Y") + '_signal.csv')
# result_df.to_csv('full_result.csv')


# if to_excel == "y":
#     modify_excel.modify()
#     logging.info("Excel Modified")

#     upload_gsheet.upload()
#     logging.info("Google Sheet Uploaded")

