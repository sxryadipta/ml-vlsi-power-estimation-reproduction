from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

def train_rf(X_train, y_train):

    rf = RandomForestRegressor(
        n_estimators=700,
        max_depth=15,
        min_samples_leaf=1,
        random_state=42
    )

    # 10-fold cross validation (as used in the paper)
    scores = cross_val_score(
        rf,
        X_train,
        y_train,
        cv=10,
        scoring="neg_mean_squared_error"
    )

    mse = -scores.mean()
    rmse = np.sqrt(mse)

    rf.fit(X_train, y_train)

    return rf, mse, rmse