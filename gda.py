# gaussian discriminant analysis

from readattributes import read_data , save_plots

import numpy as np

import matplotlib.lines as lines

from numpy import linalg as la

import matplotlib.patches as plotpatches

import matplotlib.pyplot as plt
import sys 

filex = sys.argv[1]
filey = sys.argv[2]
partnumber = sys.argv[3]

x, y , m, n = read_data(filex, filey ,"  ",'str')

alaska = x[np.where(y == "Alaska")]
canada = x[np.where(y == "Canada")]


# ques 4 part (a)

def lineargda(show = True):
	mu0 = np.mean(alaska, axis =0)
	mu1 = np.mean(canada, axis =0)
	mu = [mu0,mu1]
	
	alaska_mu0 = alaska - mu0
	canada_mu1 = canada - mu1
	
	x_mu = np.concatenate((alaska_mu0, canada_mu1), axis =0)
	
	transposex_mu = np.transpose(x_mu)
	sigma = (transposex_mu @ x_mu ) / m
	if(show):
		print("\n Part A \n")
		print("Mu_0 =", mu[0])
		print("Mu_1 =", mu[1])
		print("Sigma =", sigma)
	return mu, sigma


# ques 4 part (b)
def plotdata():
	
	plt.scatter(alaska[:,0], alaska[:, 1], marker='o',color = 'b')
	plt.scatter(canada[:,0], canada[:, 1], marker='x', color = 'r')
	classalaska = plotpatches.Patch(color ='b' , label = "Alaska")
	classcanada = plotpatches.Patch(color = 'r' , label = "Canada")
	plt.title("Plot of training data")
	#plt.legend(handles=[classalaska, classcanada])
	plt.legend(handles=[classalaska, classcanada])
	save_plots(plt, "ques4_part(b).png")
	plt.show()


# ques 4 part (c) --> http://www-scf.usc.edu/~csci567/05-LDA.pdf --> page 19

def plotlinearboundary(mu,sigma, show = True):
	
	sigma_inverse = la.pinv(sigma)
	
	# parameters of line equation : Ax = B

	tempmu = np.transpose(mu[0] - mu[1])

	A = 2 * tempmu @ sigma_inverse
	
	B = np.transpose(mu[0]) @ sigma_inverse @ mu[0] - np.transpose(mu[1]) @ sigma_inverse @ mu[1] - 2 * np.log(len(canada) / len(alaska))
	
	# plot data points
	plt.scatter(alaska[:,0], alaska[:, 1], marker='o',color = 'b')
	plt.scatter(canada[:,0], canada[:, 1], marker='x', color = 'r')
	classalaska = plotpatches.Patch(color ='b' , label = "Alaska")
	classcanada = plotpatches.Patch(color = 'r' , label = "Canada")

	# plotting line
	x0 = x[:, 0] 
	x1 = x[:, 1]
	x1 = (B - A[0] * x0) / (A[1])
	if(show):
		line, = plt.plot(x0, x1, color = "g", label="Decision Boundary")

		plt.legend(handles = [classalaska, classcanada, line])
		plt.xlabel(r'Feature 0 ($X_0)$')
		plt.ylabel(r'Feature 1 ($X_1)$')

		plt.title("Linear decision boundary")
		save_plots(plt, "ques4_part(c).png")
		plt.show()
	return x0,x1


# ques 4 part (d)

def quadraticgda():
	mu0 = np.mean(alaska, axis =0)
	mu1 = np.mean(canada, axis =0)
	mu = [mu0,mu1]
	
	alaska_mu0 = alaska - mu0
	canada_mu1 = canada - mu1

	alaska_mu0_transpose = np.transpose(alaska_mu0)
	canada_mu1_transpose = np.transpose(canada_mu1)
	sigma =  [(alaska_mu0_transpose @ alaska_mu0) / len(alaska),(canada_mu1_transpose @ canada_mu1) / len(canada)]


	print("\n Part D \n")
	print("Mu_0 =", mu[0])
	print("Mu_1 =", mu[1])
	print("Sigma0 =", sigma[0])
	print("Sigma1 =", sigma[1])
	return mu, sigma


# ques 4 part (e)
def plotquadgda(mu, sigma , line0, line1):
	print("Part E")
	# plot data points
	plt.scatter(alaska[:,0], alaska[:, 1], marker='o',color = 'b')
	plt.scatter(canada[:,0], canada[:, 1], marker='x', color = 'r')
	classalaska = plotpatches.Patch(color ='b' , label = "Alaska")
	classcanada = plotpatches.Patch(color = 'r' , label = "Canada")

	# plotting line
	line, = plt.plot(line0, line1, color = "g", label="Linear Decision Boundary")

	sigma_inverse0 = la.pinv(sigma[0])
	sigma_det0 = la.det(sigma[0])
	sigma_inverse1 = la.pinv(sigma[1])
	sigma_det1 = la.det(sigma[1])

	# computing parameters of equation x'Ax + BX + C = 0
	A = sigma_inverse0 - sigma_inverse1
	mu_transpose0 = np.transpose(mu[0])
	mu_transpose1 = np.transpose(mu[1])
	B = -2 * (mu_transpose0 @ sigma_inverse0 - mu_transpose1 @ sigma_inverse1 )
	C = (mu_transpose0 @ sigma_inverse0 @ mu[0] - mu_transpose1 @ sigma_inverse1 @ mu[1] - 2* np.log((len(canada) / len(alaska)) * (sigma_det0 / sigma_det1)) )

	#create a mesh using numpy
	T0,T1 = np.mgrid[-5:5:50j,-7.5:7.5:50j]
	xt= T0.flatten()
	yt = T1.flatten()
	mesh = np.c_[xt,yt]

	# quadratic boundary
	quadboundary = np.array([])
	for point in mesh:
		value = np.transpose(point) @ A @ point + B @ point + C
		quadboundary = np.append(quadboundary, [value])

	quadboundary = quadboundary.reshape(T0.shape)
	plt.contour(T0, T1, quadboundary, [0], colors="y")

	quadlabel = lines.Line2D(color ='yellow' , label = "quadratic decision boundary" , xdata = [] , ydata = [])

	plt.title("Quadratic Gaussian Discriminant analysis")
	plt.xlabel("Input Feature(x1)")
	plt.ylabel("Input Feature(x2)")
	
	plt.legend(handles = [classalaska, classcanada, line, quadlabel])
	save_plots(plt, "ques4_part(e).png")
	plt.show()


if(partnumber == "0"):
	mu, sigma = lineargda()
	plotdata()
	line0 , line1 = plotlinearboundary(mu,sigma)
elif(partnumber == "1"):
	mu, sigma = lineargda(False)
	line0 , line1 = plotlinearboundary(mu,sigma, False)
	qmu, qsigma = quadraticgda()
	plotquadgda(qmu, qsigma, line0, line1)


