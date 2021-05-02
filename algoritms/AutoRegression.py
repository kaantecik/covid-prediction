"""
170201015 - Erdem Özer
"""
from statsmodels.tsa.ar_model import AR


def auto_regression(df):
    train_data = df[:370]
    check_data = df[370:462]
    model = AR(train_data)
    model_fitted = model.fit()
    forecast_data = model_fitted.predict(start=370, end=461).astype(int)
    return check_data, forecast_data
