"""
180201062 - Yekta Anıl Çiçek
"""
from statsmodels.tsa.arima.model import ARIMA


def arima(df):
    data = df.values
    size = int(len(data) * 0.8)
    train, test = data[0:size], data[size:len(data)]
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        y_hat = output[0]
        predictions.append(int(y_hat))
        obs = test[t]
        history.append(obs)
    return test, predictions
