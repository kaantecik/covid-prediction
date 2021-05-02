"""
170201041
Oğuzhan Taşımaz
"""

import numpy as np
from sklearn import svm


def svr(df):
    train_data = df[:370]
    check_data = df[370:462]
    cases = np.array([])
    check_data_as_array = np.array([])
    dates = np.array(list(range(1, 371))).reshape(-1, 1)
    check_dates = np.array(list(range(370, 462))).reshape(-1, 1)

    for case in train_data:
        cases = np.append(cases, case).astype(int)

    for data in check_data:
        check_data_as_array = np.append(check_data_as_array, data).astype(int)

    regr = svm.SVR(kernel='linear')
    regr.fit(dates, cases)
    forecast_data = regr.predict(check_dates)
    return check_data_as_array.astype(int), forecast_data.astype(int)
