import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


class DataFormatter(object):
    @staticmethod
    def get_data(country=None, path=None):
        df = pd.read_csv(path)
        df = df.drop(columns=["Province/State", "Lat", "Long"])
        df = (df.melt(id_vars=['Country/Region'],
                      var_name='Date', value_name='Cases').assign(Date=lambda x: pd.to_datetime(x['Date'])))

        index = pd.date_range(start='2020-1-22', end='2021-4-27', freq="D")
        if country:
            df = df.loc[df['Country/Region'] == country]
            df = df.reset_index()
            df = df.drop(columns=["index"])
            df = pd.Series([value['Cases']
                            for key, value in df.iterrows()], index=index)
            return df

        else:
            data = []
            for date in index:
                case = int(df.loc[df['Date'] == date].sum(axis=0).values[1])
                data.append(case)

            total_cases = pd.Series(data, index=index)
            return total_cases

    @staticmethod
    def mse(actual, predicted):
        difference_array = np.subtract(actual, predicted)
        squared_array = np.square(difference_array)
        mse = squared_array.mean()
        return mse

    @staticmethod
    def mae(actual, predicted):
        return mean_absolute_error(actual, predicted)

    @staticmethod
    def r_square(actual, predicted):
        correlation_matrix = np.corrcoef(actual, predicted)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        return round(r_squared, 5)

    @staticmethod
    def draw(data, forecast):
        plt.figure(figsize=(12, 8))
        plt.plot(data, marker='o', color='black')
        plt.plot(forecast, marker='o', color='blue')
        line1, = plt.plot(data, marker='o', color='black')
        line2, = plt.plot(forecast, marker='o', color='blue')
        plt.legend([line1, line2], ['Test', 'Forecast'])
        plt.show()
