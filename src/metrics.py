import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    error_percent = abs((y_test - pred)/y_test)*100

    return pred, mse, rmse, r2, error_percent