"""
170201082
Ayris Gürbulak
"""
from numpy import mean


def moving_average(df):
    data = df.values
    window = 3
    check_data = [data[i] for i in range(window)]
    test = [data[i] for i in range(window, len(data))]
    forecast_data = list()
    for t in range(len(test)):
        length = len(check_data)
        yhat = mean([check_data[i] for i in range(length - window, length)])
        obs = test[t]
        forecast_data.append(int(yhat))
        check_data.append(obs)
    return check_data, forecast_data
