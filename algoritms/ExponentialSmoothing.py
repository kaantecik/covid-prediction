"""
170201069
Kaan Tecik
"""
import warnings
from statsmodels.tsa.api import Holt
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)


def exponential_smoothing(df, smoothing_trend=0.6, initialization_method="estimated"):
    train_data = df[:370]
    check_data = df[370:462]
    day_of_forecast = 92
    model = Holt(train_data, initialization_method=initialization_method)
    fit2 = model.fit(smoothing_trend=smoothing_trend)
    forecast = fit2.forecast(day_of_forecast).astype(int)
    return check_data, forecast
