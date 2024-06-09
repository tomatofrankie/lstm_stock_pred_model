# stock_prediction
 
## multivariate_multistep_lstm_20220706_mayoverfit.ipynb
- Two layer LSTM (32),(16)
- Close, macd, macds
- Result: Overfit

## multivariate_directional_lstm_20230916.ipynb
- One layer LSTM (32)
- binary_crossentropy
- Result: Low accuarcy

## stacked_lstm_20230925.ipynb
- Three stacked layer LSTM (32),(16),(8)
- Close, High, Low
- Reference: Ding, G., Qin, L. Study on the prediction of stock price based on the associated network model of LSTM. Int. J. Mach. Learn. & Cyber. 11, 1307–1317 (2020). https://doi.org/10.1007/s13042-019-01041-1
- Result: Low accuracy

## stacked_directional_lstm_20230925.ipynb
- Three stacked layer LSTM (256),(128),(64)
- Close, High, Low
- Result: Low accuracy

## multivariate_multistep_lstm_20231006.ipynb
- Two layer LSTM (32),(16)
- Close, atr
- Result: No significant improvement on current model

## multivariate_multistep_var_20231006.ipynb
- VAR
- Close, macd, macds
- Result: Underfit

## Contact
Instagram: [algolutionhk](https://www.instagram.com/algolutionhk/)  
Telegram: [algolutionhk](https://t.me/algolutionhk)  
LinkedIn: [Algolution HK](https://www.linkedin.com/company/algolutionhk/)  
Email: [algolutionhk@gmail.com](mailto:algolutionhk@gmail.com)

## Library Used
scikit-learn, yfinance, technical-indicators, statsmodels, keras 

## Reference
CNN-LSTM: https://link.springer.com/article/10.1007/s00530-021-00758-w  
Directional LSTM: https://ieeexplore.ieee.org/abstract/document/7966019  
Stacked: https://link.springer.com/article/10.1007/s13042-019-01041-1  

Bi-LSTM: https://ieeexplore.ieee.org/abstract/document/8355458  
Bi-LSTM: https://ieeexplore.ieee.org/abstract/document/9257950  

Multivariate LSTM:  
https://www.kaggle.com/code/bagavathypriya/multivariate-time-series-analysis-using-lstm  
https://www.kaggle.com/code/pauldavid22/multivariate-time-series-using-lstms  
https://www.kaggle.com/code/nikitricky/multivariate-multi-step-time-series-forecasting 