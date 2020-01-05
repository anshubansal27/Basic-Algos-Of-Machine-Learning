import os
import numpy as np

# to save the plot figure in the output directory

def save_plots(plot, name):
	# output directory to save the plots

	OUT = "output/"

	# creating output dorectory if not exist

	if not os.path.exists(OUT):
		os.makedirs(OUT)
	figure = os.path.join(OUT, name)
	plot.savefig(figure)


# normalize the column of numpy array
def normalize(x):

	mu , std = np.mean(x) , np.std(x)
	norm = (x-mu)/std
	return norm


# read data from the files
def read_data(xfile, yfile, delimeter = ",", datatype = None):
	x = np.loadtxt(xfile, delimiter = delimeter)
	try:
		m, n = x.shape[0], x.shape[1]
	except IndexError :
		m,n = x.shape[0], 1
	x0= np.ones(m);
	if(n ==1 ):
		x = normalize(x);
		xf = np.c_[x0,x]
	elif( n==2) :
		x1 = normalize(x[:, 0])
		x2 = normalize(x[:, 1])
		xf = np.c_[x0,x1,x2]

	if datatype is None:
		y = np.loadtxt(yfile)
	else:
		y= np.loadtxt(yfile, datatype)
		xf = np.c_[x1,x2]


	return xf,y,m,n

		




