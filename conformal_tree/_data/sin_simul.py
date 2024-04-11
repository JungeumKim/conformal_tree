import numpy as np
from functools import partial

def generate_data(n, p, cond_exp, noise_sd_fn, 
                  x_dist = partial(np.random.uniform, low=0, high=10),
                  x_option=1, noise_option=1):
    x = x_dist(size=n*p).reshape(n,p)
    noise_sd = noise_sd_fn(x,option=noise_option)
    noise = np.random.normal(scale=noise_sd, size=n)
    y = cond_exp(x,option=x_option)+noise
    return x,y

def cond_exp(x, option=1):
    if option==1:
        return np.sin(1/(x[:,0]**3))
    else:
        return 3*np.sin(4/(x[:,0]+0.2))+1.5

def noise_sd_fn(x,option=1):
    if option==1:
        return 1*x[:,0]**2
    else:
        return .1

def get_data(n0 = 50, n=100,T=800, seed = 1234,p=1 ,x_option=1,noise_option=1):
    np_random = np.random.RandomState(seed)
    x_dist = partial(np_random.beta, a=1.2, b=0.8)
    
    assert n0<n

    #n0 = int(n/2) # number of training points

    data = generate_data(n+T, p, cond_exp, noise_sd_fn, x_dist, 
                         x_option=x_option,noise_option=noise_option)
    x = data[0]
    y = data[1]

    if len(x.shape)==1:
        x = x.reshape(-1,1)

    x_train = x[:n0]
    y_train = y[:n0]

    x_calib = x[n0:n]
    y_calib = y[n0:n]

    x_test = x[n:]
    y_test = y[n:]
    
    return (x_train,y_train),(x_calib,y_calib ), (x_test, y_test)