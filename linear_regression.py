""" Ques1 Linea Regression """ 

import numpy as np

import matplotlib.pyplot as plt
from readattributes import read_data, save_plots
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  
import sys

filex = sys.argv[1]
filey = sys.argv[2]
learning_rate = float(sys.argv[3])
time_gap = float(sys.argv[4])

x, y, m, n = read_data(filex, filey)


# computing J(theta)

def J(theta):
	hxy = ((y - x @ theta) ** 2 ) / (2 * m)
	return np.sum(hxy)


# ques1 part(a)

def batchgradient(eta = 0.001):
	
	print("learning rate : " , eta)
	
	# converged initially is false
	converged = False

	# counting the number of iterations
	iterationcount = 0

	# initialise all thetas to 0 initially
	theta = np.zeros(n+1)

	# keep trace of all errors during the algorithm execution
	traceJ = np.array([])

	# computing error when all thetas =0 
	Je = J(theta)
	
	tracearr = np.array([])
	for i in range(n+1) :
		tracearr = np.append(tracearr,[theta[i]])
	tracearr = np.append(tracearr,[Je])
	

	traceJ = np.append(traceJ, tracearr)

	while( not converged) :
		#gradient descent update rule
		theta -= (eta * (x @ theta - y) @ x)/m

		prevJe = Je
		Je = J(theta)
		
		iterationcount +=1
		
		if(abs(Je - prevJe) < 10 ** (-10) ) :
			converged = True
			break
		
		# to stop the infinite loop
		if(iterationcount >= 10000) :
			print(" \n iteration count exceeded \n ")
			break

		tracearr = np.array([])
		for i in range(n+1) :
			tracearr = np.append(tracearr,[theta[i]])
		tracearr = np.append(tracearr,[Je])
	

		traceJ = np.append(traceJ, tracearr)

	print("theta Parameters  : " , theta)
	print("Stopping criteria for convergence : abs(prev error - new error) < 10 ** (-10) " )

	return theta, traceJ.reshape(iterationcount, n+2), converged


# quest 1 part (b)
def plotdata(theta):
	data, = plt.plot(x[:, 1],y, 'rx', label ="Data" )
	hy = list(map(lambda y: theta @ y, x))
	plt.xlabel("Acidity of Wine (Normalised) ")
	plt.ylabel("Density Of Wine ")

	line, = plt.plot(x[:, 1] ,hy ,"b" , label = "Hypothesis")

	plt.legend(handles = [data, line])

	save_plots(plt, "ques_1b.png")
	plt.show()



# ques 1 part(c)

def plotmesh(traceJ, timegap):

	plt.ion()
	pltfig = plt.figure()

	#create a mesh using numpy
	T0,T1 = np.mgrid[0:2:50j,-1:1:50j]
	xt= T0.flatten()
	yt = T1.flatten()
	mesh = np.c_[xt,yt]

	#meshplot = pltfig.add_subplot(111, projection = '3d')
	meshplot = pltfig.gca(projection = '3d')
	
	#compute Jvalues for the grid according to our mesh points
	Jvalue = (np.array([J(point) for point in mesh]).reshape(50,50))
	
	meshplot.set_xlabel(r'$\theta 0$', labelpad = 15)
	meshplot.set_ylabel(r'$\theta 1$', labelpad = 15)
	meshplot.set_zlabel(r'$J(\theta)$', labelpad = 15)

	meshplot.plot_surface(T0,T1, Jvalue, cmap = cm.RdBu_r)

	plt.show()
	for value in traceJ :
		meshplot.plot([value[0]],[value[1]],[value[2]] , color ='b', marker ='x', markersize =2)
		if(timegap > 0.0):
			plt.pause(timegap)

	save_plots(plt, "ques_1c.png")
	plt.close()


# ques 1 part (d)

def plotcontour(traceJ,timegap,eta,imagename, converged):
	plt.ion()
	pltfig = plt.figure()

	#create a mesh using numpy
	T0,T1 = np.mgrid[0:2:50j,-1:1:50j]
	mesh = np.c_[T0.flatten(), T1.flatten()]

	reshapevalue = T0.shape

	Jvalue = (np.array([J(point) for point in mesh]).reshape(reshapevalue))
	
	plt.xlabel(r'$\theta 0$')
	plt.ylabel(r'$\theta 1$')

	plt.contour(T0,T1, Jvalue, cmap = cm.RdBu_r)

	plt.title("Contour Plot (eta = " +str(eta) + ")")
	plt.show()
	i = 0
	for value in traceJ :
		i += 1
		plt.plot([value[0]],[value[1]] , color ='b', marker ='x', markersize =2)
		if(timegap > 0.0):
			plt.pause(timegap)
		if(not converged and i > 99):
			break

	save_plots(plt, imagename)
	plt.close()


# ques 1 part (d)

def plotcontours_eta():
	etavalues = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5]
	for eta in etavalues:
		print("for eta = %f" %eta)
		theta, traceJ, converged = batchgradient(eta)
		imagename = "ques_1e_eta_" + str(eta) + ".png" 
		plotcontour(traceJ, 0.0, eta, imagename,converged )

	
	


theta, traceJ, converged = batchgradient(learning_rate)
plotdata(theta)
plotmesh(traceJ,time_gap)
plotcontour(traceJ,time_gap,learning_rate, "ques_1d.png", converged)
plotcontours_eta()
 
		
	
	
	
	
