import numpy as np

def gen_test_data(size=200):


    X = np.sort(np.random.uniform(0,2,size))

    def f(x):
        return x*np.sin(1/x)

    noise_std = np.ones(X.shape) + 2*(X>1)
    y = f(X) + np.random.normal(0,noise_std,size)

    return X.reshape(-1,1), y

def gen_sin_data(data=1, size=500):
    X = np.random.uniform(0,10,size=500)
    if data == 1:
        mu = 3*np.sin(4/X + 0.2) + 1.5
        sd = X
    elif data == 2:
        mu = np.sin(X**(-3))
        sd = 0.1
    elif data == 3:
        mu = np.sin(X**(-3))
        sd = X
    y = np.random.normal(mu, sd)

    return X.reshape(-1,1),y
