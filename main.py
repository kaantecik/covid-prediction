from algoritms.SVR import svr
from algoritms.ARIMA import arima
from utils.DataFormatter import DataFormatter
from algoritms.MovingAverage import moving_average
from algoritms.AutoRegression import auto_regression
from algoritms.LinearRegression import linear_regression
from algoritms.ExponentialSmoothing import exponential_smoothing

data_formatter = DataFormatter()

country = "Germany"
confirmed_path = "data/time_series_covid19_confirmed_global.csv"
death_path = "data/time_series_covid19_deaths_global.csv"
recovered_path = "data/time_series_covid19_recovered_global.csv"

death_df = data_formatter.get_data(country=country, path=death_path)
confirmed_df = data_formatter.get_data(country=country, path=confirmed_path)
recovered_df = data_formatter.get_data(country=country, path=recovered_path)

if __name__ == "__main__":
    # ES
    recovered_data, recovered_forecast = exponential_smoothing(df=recovered_df)
    death_data, death_forecast = exponential_smoothing(df=death_df)
    confirmed_data, confirmed_forecast = exponential_smoothing(df=confirmed_df)

    # MA
    death_check, death_forecast = moving_average(death_df)
    confirmed_check, confirmed_forecast = moving_average(confirmed_df)
    recovered_check, recovered_forecast = moving_average(recovered_df)

    # AR
    death_check, death_forecast = auto_regression(death_df)
    confirmed_check, confirmed_forecast = auto_regression(confirmed_df)
    recovered_check, recovered_forecast = auto_regression(recovered_df)

    # LR
    death_check, death_forecast = linear_regression(death_df)
    confirmed_check, confirmed_forecast = linear_regression(confirmed_df)
    recovered_check, recovered_forecast = linear_regression(recovered_df)

    # ARIMA
    death_check, death_forecast = arima(death_df)
    confirmed_check, confirmed_forecast = arima(confirmed_df)
    recovered_check, recovered_forecast = arima(recovered_df)

    # SVR
    death_check, death_forecast = svr(death_df)
    confirmed_check, confirmed_forecast = svr(confirmed_df)
    recovered_check, recovered_forecast = svr(recovered_df)
