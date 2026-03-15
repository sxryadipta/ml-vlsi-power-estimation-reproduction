from sklearn.neural_network import MLPRegressor


def train_bpnn(X_train, y_train):

    model = MLPRegressor(
        hidden_layer_sizes=(15,15),
        activation="tanh",
        solver="lbfgs",
        max_iter=3000,
        learning_rate_init=0.5,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model