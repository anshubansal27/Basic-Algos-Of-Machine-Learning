# question 3 logistic regression - classification problem 

# ******************** https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function (for computation algo) 

import numpy as np

from numpy import linalg as la
import matplotlib.pyplot as plt

from readattributes import read_data, save_plots

import matplotlib.patches as plotpatches

import sys 

filex = sys.argv[1]
filey = sys.argv[2]

x, y, m, n = read_data(filex, filey)


# hypothesis function for logistic regression
def h(theta):
	# computing z =  x X theta
	z = x @ theta
	# computing g(z) - sigmoid function
	g = 1 / (1 + np.exp(-z))

	return g


# gradient computation for the log - likelihood function

def gradient(theta):
	transposex =  np.transpose(x)
	y_h = y- h(theta)
	return transposex @ y_h

# hessian metrix computation
def hessian(theta):
	# -> X D X(transpose) where D = g(z) * (1-g(z))
	gz = h(theta)
	transposex = np.transpose(x)
	d = np.diag(gz * (1-gz))
	return transposex @ d @ x

# log-likelihood of logistic regression
def L(theta):
	# as given on above mentioned link
	hvalue = h(theta)
	return -1 * (np.sum(y * np.log(hvalue) + (1 - y) * np.log(1 - hvalue)))
	


# ques 3 part a)

def logisticregression():
	theta = np.zeros(n+1)
	converged = False
	iterationcount =0

	nexterror = L(theta)

	while not converged:
		grad = gradient(theta)
		hessianpinv = la.pinv(hessian(theta))
		theta += hessianpinv @ grad
		initialerror = nexterror
		nexterror = L(theta)
		iterationcount += 1
		if(abs(initialerror - nexterror) < 10 ** -10):
			converged = True
	
	print("parameters : " , theta)

	return theta



# ques 3 part (b)

def plotting(theta):
	def colorslist():
		colors =[]
		for cls in y:
			if cls:
				colors.append("r")
			else :
				colors.append("b")
		return colors
	
	color = colorslist()
	plt.scatter(x[:, 1], x[:, 2], c=color)
	

	# finding x2 value using line equation --> x2 = (-theta0 -theta1 x1) / theta2 for each x
	yy = []
	for xi in x :
		value = (-theta[0] - theta[1] * xi[1]) / theta[2]
		yy.append(value)
	cls0 = plotpatches.Patch(color='blue', label='Class 0')
	cls1 = plotpatches.Patch(color='red', label='Class 1')

	line, = plt.plot(x[:, 1], yy, 'g', label="Decision Boundary")

	plt.xlabel(r'Feature 0 ($X_0)$')
	plt.ylabel(r'Feature 1 ($X_1)$')
	
	plt.legend(handles=[cls0, cls1, line])
	
	save_plots(plt, "ques3_b.png")
	plt.show()
	

theta = logisticregression()
plotting(theta)

	
