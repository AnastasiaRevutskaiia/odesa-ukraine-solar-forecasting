import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from os import walk
import numpy as np
from datetime import datetime, timedelta
import re
from textwrap import wrap

BASE_FOLDER_WITH_PREDICTIONS = '/Users/citrus/Desktop/solar_forecasting-main/solar_rad/'

whitelist = ['1-arima', '1-dbn', '1-elm', '1-gb', '1-het', '1-mlp', '1-rf', '1-svr']


def is_in_whitelist(name):
    for allowedName in whitelist:
        if name.find(allowedName) != -1:
            return True
    return False


def read_all_results():
    filenames = []
    folders = []

    for (root, dirs, files) in walk(BASE_FOLDER_WITH_PREDICTIONS):
        for folder in dirs:
            if is_in_whitelist(folder):
                folders.append(folder)

        for file in files:
            if is_in_whitelist(file):
                filenames.append(file)

    return folders, filenames


def get_results_for_predictions():
    folderNames, filenames = read_all_results()

    predictions = np.empty(shape=[0, 2])
    for folderName in folderNames:
        for filename in filenames:
            if filename.find(folderName) != -1:
                filepath = BASE_FOLDER_WITH_PREDICTIONS + folderName + '/' + filename
                predictionFile = open(filepath, 'rb')
                predictions = np.append(predictions, [[filepath, pickle.load(predictionFile)]], axis=0)
                predictionFile.close()

    return predictions


def find_best_model_for_prediction(predictions):
    RMSEs = []
    for predictionResult in predictions[:, 1]:
        if predictionResult.get('test_metrics').get('RMSE') != 0:
            RMSEs.append(predictionResult.get('test_metrics').get('RMSE'))

    minRMSE = min(RMSEs)

    for prediction in predictions:
        if prediction[1].get('test_metrics').get('RMSE') == minRMSE:
            return prediction

    raise Exception('Unable to identify model with best prediction power')


def daterange(end_date, lookbackHours):
    daterange = []
    delta = timedelta(hours=1)
    start_date = end_date - timedelta(hours=(lookbackHours - 1))
    while start_date <= end_date:
        daterange.append(start_date)
        start_date += delta
    return daterange


def extract_title_for_prediction(prediction):
    try:
        return re.search('\/1-(.*){1}\/.*.pkl', prediction[0]).group(1)
    except AttributeError:
        return 'Unable to parse title for model'


def plot_prediction(prediction, lookbackHours=48):
    end_date = datetime(2022, 7, 9, 23, 00)
    dateRangeOfPredictions = daterange(end_date, lookbackHours)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    startIndexOfPredictions = len(prediction[1]['real_values']) - lookbackHours

    labelReal, = plt.plot(dateRangeOfPredictions, prediction[1]['real_values'][startIndexOfPredictions:], color='r',
                          label='real values')
    labelPredicted, = plt.plot(dateRangeOfPredictions, prediction[1]['predicted_values'][startIndexOfPredictions:],
                               color='g',
                               label='predicted values')
    plt.gcf().autofmt_xdate()
    plt.legend(handles=[labelReal, labelPredicted])
    plt.xlabel("Date per hourly resolution")
    plt.ylabel("Global Horizontal Irradiance (GHI)- W/m2")
    title = extract_title_for_prediction(prediction)
    plt.title(wrap(title, 50))
    plt.show()


if __name__ == '__main__':
    predictions = get_results_for_predictions()
    # bestModelForPrediction = find_best_model_for_prediction(predictions)
    # plot_prediction(bestModelForPrediction)
    #
    for prediction in predictions:
        plot_prediction(prediction)
