# Solar Forecasting
The initial data of solar irradiance was retrieved via resource: https://toolkit.solcast.com.au/historical

# Usage
General information:
- There is a jupyter notebook to execute every single model (TODO: ARIMA execution);
- After executing all single model, is possible to use the notebook called by heterogen_ensemble.ipynb, to create the Dynamic Selection experiments;

Files information:
- The file models_configuration_60_20_20.json is responsible to configure the time series paths, test and validation splits, series names, and time series codes;
- All time-series data are inside of ./data/;
- All models execution will be persisted in ./solar_rad/;
- The ARIMA model are persisted in ./arima_models/;

Python files description:
- fit_predict_models.py: responsible for training and predicting scripts for Sklearn based models;
- heterogen_ensemble.py: responsible for creating dynamic selection approaches;
- time_series_functions.py: scripts for persisting and load results, and compute performance metrics;


# No module named 'sklearn' Error ?
```
pip3 install -U scikit-learn scipy matplotlib
```
