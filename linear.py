import math
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#import seaborn as sns

from scipy import linalg
def ridge(A, b, alphas):
    """
    Return coefficients for regularized least squares

         min ||A x - b||^2 + alpha ||x||^2

    Parameters
    ----------
    A : array, shape (n, p)
    b : array, shape (n,)
    alphas : array, shape (k,)

    Returns
    -------
    coef: array, shape (p, k)
    """
    U, s, Vt = linalg.svd(A, full_matrices=False)
    print('USV = ', U.shape, s.shape, Vt.shape)
    d = s / (s[:, np.newaxis].T ** 2 + alphas[:, np.newaxis])
    return np.dot(d * U.T.dot(b), Vt).T


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()
    
def predict(X, W):
    Xpred = np.c_[np.ones(X.shape[0]),X]
    return np.dot(X, W)
        
# linear regression using "mini-batch" gradient descent 
# function to compute hypothesis / predictions 
def hypothesis(X, theta): 
	return np.dot(X, theta) 

# function to compute gradient of error function w.r.t. theta 
def gradient(X, y, theta, alpha): 
	h = hypothesis(X, theta) 
	grad = np.dot(X.transpose(), (h - y)) + alpha * theta
	return grad 

# function to compute the error for current values of theta 
def cost(X, y, theta): 
	h = hypothesis(X, theta) 
	J = np.dot((h - y).transpose(), (h - y)) 
	N = len(y)
	#print('N = ', N)
	J /= N  #*np.ptp(y)
	#print('h = ', h)
	return J[0] 
	
def nrmse(X, y, theta): 
	h = hypothesis(X, theta) 
	J = np.dot((h - y).transpose(), (h - y)) 
	N = len(y)
	J /= N*np.ptp(y)  # ptp - peak to peak range
	return J[0] 

# function to create a list containing mini-batches 
def create_mini_batches(X, y, batch_size): 
	mini_batches = [] 
	data = np.hstack((X, y)) 
	np.random.shuffle(data) 
	n_minibatches = data.shape[0] // batch_size 
	i = 0

	for i in range(n_minibatches + 1): 
		mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
		X_mini = mini_batch[:, :-1] 
		Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
		mini_batches.append((X_mini, Y_mini)) 
	if data.shape[0] % batch_size != 0: 
		mini_batch = data[i * batch_size:data.shape[0]] 
		X_mini = mini_batch[:, :-1] 
		Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
		mini_batches.append((X_mini, Y_mini)) 
	return mini_batches 

# function to perform mini-batch gradient descent 
def gradientDescent(X, y, learning_rate = 0.00001, batch_size = 32, alpha = 0.1): 
	theta =  np.zeros((X.shape[1], 1)) #/ 1000
	error_list = [] 
	max_iters = 300
	beta = 0.9  # momentum coef
	v =  np.zeros((X.shape[1], 1)) # momentum value
	cnt = 2
	for itr in range(max_iters): 
		print(cnt)
		learning_rate = learning_rate/cnt
		cnt = cnt + 1
		mini_batches = create_mini_batches(X, y, batch_size) 
		for mini_batch in mini_batches: 
			X_mini, y_mini = mini_batch 
			grad = gradient(X_mini, y_mini, theta, alpha) 
			#print('len grad = ', len(grad))
			v = beta * v + (1 - beta) * grad 
			theta = theta - learning_rate * v
			error_list.append(nrmse(X_mini, y_mini, theta)) 

	return theta, error_list 


## Read dataset
N_Ntr = [] 
header = {0,1} 
for i, row in enumerate(open('2.txt')):
    if i in header: 
        N_Ntr.append(row.strip('\n')) 

N = int(N_Ntr[0])  # features num
Ntr = int(N_Ntr[1])  # train dataset num
print(N,Ntr) 

N_test = [] 
test_num_line = {Ntr+2} 
for i, row in enumerate(open('2.txt')):
    if i in test_num_line: 
        N_test.append(row.strip('\n')) 

Nts = int(N_test[0])  # test dataset num
print(Nts)

with open('2.txt') as f:
    data = f.readlines()

train_lines = data[2:Ntr+2]
train0 = []
train = np.zeros((Ntr, N+1))
for i in range(len(train_lines)):
    train0 = list(map(int, train_lines[i].split()))
    train[i,:] = np.asarray(train0)
    
print(train.shape)

test_lines = data[Ntr+3:]
test0 = []
test = np.zeros((Nts, N+1))
for i in range(len(test_lines)):
    test0 = list(map(int, test_lines[i].split()))
    test[i,:] = np.asarray(test0)
    
print(test.shape)

X_train = train[:, :-1]
#X_train = (X_train - np.mean(X_train))/np.std(X_train)
y_train = train[:, -1].reshape((-1, 1)) 

X_test = test[:, :-1]
#X_test = (X_test - np.mean(X_train))/np.std(X_train)
y_test = test[:, -1].reshape((-1, 1)) 

#print(X_train)
## Ridge regression
#alphas = [0.7, 0.8, 0.9]
#coeffs = ridge(X_train, y_train, alphas)

'''
n_pnts = 13
x_exp = np.zeros(n_pnts)
ridge_nrmse = np.zeros(n_pnts)
for i in range(n_pnts):
	x_exp[i] = 100000/(10**i)
	ridge = Ridge(alpha = x_exp[i], fit_intercept=False, normalize = False, solver = 'svd')
	ridge.fit(X_train, y_train)
	#y_train_pred = predict(X_train, ridge.coef_.transpose())
	#ridge_nrmse[i] = np.sqrt(mse(y_train, y_train_pred)) #/np.ptp(y_train) ############################################
	y_test_pred = predict(X_test, ridge.coef_.transpose())
	ridge_nrmse[i] = np.sqrt(mse(y_test, y_test_pred))

print(np.flip(x_exp), np.flip(ridge_nrmse)) 
plt.plot(np.flip(x_exp), np.flip(ridge_nrmse)) 
plt.xlabel("alpha") 
plt.ylabel("NRMSE") 
plt.show()
	
ridge = Ridge(alpha = 0.0000001, fit_intercept=False, normalize = False, solver = 'svd')
ridge.fit(X_train, y_train)             # Fit a ridge regression on the training data
y_train_pred = ridge.predict(X_train)           # Use this model to predict the test data
#print(pd.Series(ridge.coef_, index = X.columns)) # Print coefficients
print('Train MSE = ', mean_squared_error(y_train, y_train_pred))          # Calculate the test MSE
print('my Train MSE = ', mse(y_train, y_train_pred)) 
print('Train cost = ', cost(X_train, y_train, ridge.coef_.transpose()))
print('Coeffs len = ', len(ridge.coef_.transpose()))

#y_test_pred = ridge.predict(X_test)           # Use this model to predict the test data
y_test_pred = predict(X_test, ridge.coef_.transpose()) 
print('Test MSE = ', mean_squared_error(y_test, y_test_pred))          # Calculate the test MSE
print('my Test MSE = ', mse(y_test, y_test_pred)) 
print('Test cost = ', cost(X_test, y_test, ridge.coef_.transpose()))


'''
theta, error_list = gradientDescent(X_train, y_train) 
print("Bias = ", theta[0]) 
#print("Coefficients = ", theta[1:]) 

# visualising gradient descent 
print('error len = ', len(error_list))
print(error_list[:10])  # initial elements
print(error_list[-10:])  # ending elements
plt.plot(error_list[1:20])
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()


