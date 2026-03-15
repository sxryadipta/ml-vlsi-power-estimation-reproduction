import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(train_path, test_path):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(["circuit","power"], axis=1)
    y_train = train["power"]

    X_test = test.drop(["circuit","power"], axis=1)
    y_test = test["power"]

    scaler = MinMaxScaler(feature_range=(-1,1))

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test