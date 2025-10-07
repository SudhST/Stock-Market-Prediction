from sklearn.metrics import mean_squared_error

def train_and_evaluate(model, X_train, y_train, X_test, y_test, fit_params={}):
    model.fit(X_train, y_train, **fit_params)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse, preds
