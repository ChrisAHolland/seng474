import numpy as np
import matplotlib.pyplot as plt

# GPA, GRE scores dataset

X=np.array([
[1,1.0,1.0],
[1,0.9,1.0],
[1,0.9,0.875],
[1,0.7,0.75],
[1,0.6,0.875],
[1,0.6,0.875],
[1,0.5,0.75],
[1,0.5,0.8125],
[1,0.5,1.0],
[1,0.5,0.875],
[1,0.5,0.875]])

print(X)

y=np.array([[
1,
1,
1,
-1,
-1,
1,
-1,
-1,
1,
-1,
1
]]).T;

print(y)

w = np.ones((1,X.shape[1]))

#TODO
def error(x,y,w):
    return np.log(1 + np.exp(-y * x@w.T))

#TODO
def error_mean(X,y,w):
    n = X.shape[0]
    data = error(X, y, w)
    data = np.sum(data)
    return data/n

print(error_mean(X,y,w))

#TODO
def grad(x,y,w):
    return (y * x)/(1+np.exp(y * x@w.T))

#TODO
def grad_mean(X,y,w):
    n = X.shape[0]
    data = grad(X,y,w)
    data = np.sum(data, axis=0, keepdims=True)
    return (-1/n)*data

print(grad_mean(X,y,w))

def fit(X,y,kappa,iter):
    w = np.zeros((1,X.shape[1]))
    E = []

    #TODO
    n = X.shape[0]
    for i in range(0, iter):
        grad = grad_mean(X, y, w)
        w = w - kappa*grad
        E.append(error_mean(X, y, w))
    return w,E

w,E = fit(X,y,1,100)
print(w)
plt.plot(E)
plt.show()