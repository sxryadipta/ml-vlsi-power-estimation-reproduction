from preprocess import load_data
from bpnn_model import train_bpnn
from rf_model import train_rf
from metrics import evaluate

X_train, X_test, y_train, y_test = load_data(
    "../data/train.csv",
    "../data/test.csv"
)

bpnn = train_bpnn(X_train, y_train)
rf, rf_mse_cv, rf_rmse_cv = train_rf(X_train, y_train)

bp_pred, bp_mse, bp_rmse, bp_r2, bp_err = evaluate(bpnn, X_test, y_test)
rf_pred, rf_mse, rf_rmse, rf_r2, rf_err = evaluate(rf, X_test, y_test)

print("BPNN")
print(bp_mse, bp_rmse, bp_r2)

print("Random Forest")
print(rf_mse, rf_rmse, rf_r2)