"""
170201041
Oğuzhan Taşımaz
"""
import numpy as np
from sklearn import linear_model


def linear_regression(df):
    train_data = df[:370]
    check_data = df[370:462]

    cases = np.array([])
    check_data_as_array = np.array([])

    dates = np.array(list(range(1, 371))).reshape(-1, 1)
    check_dates = np.array(list(range(370, 462))).reshape(-1, 1)

    for case in train_data:
        cases = np.append(cases, case)

    for data in check_data:
        check_data_as_array = np.append(check_data_as_array, data)

    y_train = np.array([cases]).reshape(-1, 1)
    y_test = np.array([check_data_as_array]).reshape(-1, 1)

    x_train = np.array([dates]).reshape(-1, 1)
    x_test = np.array([check_dates]).reshape(-1, 1)

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)
    forecast_data = y_pred

    return check_data_as_array.astype(int), forecast_data.astype(int)
