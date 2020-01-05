# ques2 Weighted regression

from numpy import linalg as la

import numpy as np

from readattributes import read_data, save_plots

import matplotlib.pyplot as plt
import sys 

filex = sys.argv[1]
filey = sys.argv[2]
tauparameter = float(sys.argv[3])

x, y, m, n = read_data(filex, filey)


# ques1 part(a)

def unweightedregression():

	# computing theta using normal equation
	transposex = np.transpose(x)
	inversematrix = la.inv(transposex @ x)
	theta = inversematrix @ transposex @ y

	print ( "theta parameters : ", theta, "\n")
	hy = list(map(lambda y: theta @ y, x))

	line, = plt.plot(x[:, 1] ,hy ,"b" , label = "Hypothesis")

	data, = plt.plot(x[:, 1],y, 'rx', label ="Data" )

	plt.legend(handles = [data, line])

	plt.xlabel("input x")
	plt.ylabel("output y")

	save_plots(plt, "ques_2a.png")
	plt.show() 


# ques2 part (b)

def weightedregression(tau,imagename):

	#generating points for plot
	maxx = np.amax(x[:, 1])
	minx = np.amin(x[:, 1])
	pointsforplot = np.linspace(minx,maxx)

	result = np.array([])

	for point in pointsforplot :
		w = np.exp(-1 * ((point - x[:, 1])**2 / (2 * tau ** 2)))
		W = np.diag(w)
		transposex = np.transpose(x)
		inversematrix = la.inv(transposex @ W @ x)
		theta = inversematrix @ transposex @ W @ y
		result = np.append(result, [theta @ np.array([1, point])])

	# plot of the result obtained from weighted regression
	
	line, = plt.plot(pointsforplot, result, 'b', label="Hypothesis for " + r'$\tau =  ' + str(tau))
	data, = plt.plot(x[:, 1],y, 'rx', label ="Data" )
	plt.xlabel("input x")
	plt.ylabel("output y")

	plt.legend(handles = [data, line])

	plt.title("for value tau = " + str(tau))

	save_plots(plt, imagename)
	plt.show() 


#ques 2 part (c)

def weightedregression_tau():
	tauvalues =[0.1, 0.3, 2, 10]
	for tau in tauvalues:
		imagename = "ques_2c_tau_" + str(tau) + ".png"
		weightedregression(tau, imagename)

	
		

	
	



unweightedregression()
weightedregression(tauparameter,"ques_2b.png")
weightedregression_tau()

	
	
	

