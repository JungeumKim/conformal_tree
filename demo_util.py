import numpy as np

def gen_sin_data(data=1, n_train=1000, n_calib=500, n_test=500):
    X = np.random.uniform(0,1,size=n_train+n_calib+n_test)
    if data == 1:
        mu = 3*np.sin(4/(X + 0.2)) + 1.5
        sd = X
    elif data == 2:
        mu = np.sin(X**(-3))
        sd = 0.1
    elif data == 3:
        mu = np.sin(X**(-3))
        sd = X
    y = np.random.normal(mu, sd)

    X = X.reshape(-1,1)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_calib = X[n_train:(n_train+n_calib)]
    y_calib = y[n_train:(n_train+n_calib)]
    X_test = X[n_train+n_calib:]
    y_test = y[n_train+n_calib:]

    return X_train, y_train, X_calib, y_calib, X_test, y_test


def split_conformal(y_calib, y_calib_pred, y_test_pred, alpha):
    scores = np.abs(y_calib - y_calib_pred)
    n = len(y_calib)
    gap = np.quantile(scores, np.ceil((1-alpha)*(n+1))/n)
    return y_test_pred - gap, y_test_pred + gap
