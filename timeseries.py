from __future__ import print_function
from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from sys import stdout
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import sklearn as sk 
import pickle
import math

active_max = 0
sub1_max = 0
sub2_max = 0
sub3_max = 0

def train (ds, net):
	# Train the network 
	trainer = RPropMinusTrainer(net, dataset=ds)
	train_errors = [] # save errors for plotting later
	EPOCHS_PER_CYCLE = 5
	CYCLES = 100
	EPOCHS = EPOCHS_PER_CYCLE * CYCLES
	for i in xrange(CYCLES):
	    trainer.trainEpochs(EPOCHS_PER_CYCLE)
	    error = trainer.testOnData()
	    train_errors.append(error)
	    epoch = (i+1) * EPOCHS_PER_CYCLE
	    print("\r epoch {}/{}".format(epoch, EPOCHS))
	    stdout.flush()

	# print("final error =", train_errors[-1])

	return train_errors, EPOCHS, EPOCHS_PER_CYCLE

def predict (ds, net, date):
    #predict on test dataset
	i = 0 
	sam = []
	tar = []
	filename = "Result.csv"
	f = open(filename, 'w')
	f.write('DateTime, Active, Predicted next Active, Actual Next Active \n')
     #write to csv
	for sample, target in ds.getSequenceIterator(0):
		s = '{0},{1},{2}, {3}\n'.format(date[i], (sample * active_max), (net.activate(sample)* active_max), (target * active_max))
		sam.append(net.activate(sample)* active_max)
		tar.append(target * active_max)
		print (s)
		f.write(s)
		i += 1
  
	print ("Created " + filename)
	f.close()
	return sam, tar

def ploterror (train_errors, EPOCHS, EPOCHS_PER_CYCLE):
    #plot the error rate 
	plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
	plt.xlabel('epoch')
	plt.ylabel('error')
	plt.hold('on')
	plt.savefig( 'train_error', fmt='png', dpi=100, bbox_inches='tight' )   
	plt.show()
 
def plot (sam, tar):
    #plot the actual value and predicted value
    plt.subplots(figsize=(14,10))
    plt.suptitle('prediction')
    plt.plot(range(0, len(sam)), sam, c='green', label='prediction')
    plt.plot(range(0, len(tar)), tar, c='red', label='target')
    plt.xlabel('Time')
    plt.ylabel('active consumption')
    plt.hold('on')
    plt.savefig( 'prediction', fmt='png', dpi=100, bbox_inches='tight' )
    plt.show()

def create_train_set (consumption):
    #create train/test set
	global active_max 
	ds = SequentialDataSet(1, 1)
	consumption_data = normalize (consumption) 
	active_max = max(consumption_data[1],active_max)
	consumption = consumption_data[0]

	size = len (consumption)
	for i in range(0, size-1):
		ds.addSample(consumption[i], consumption[i+1])	

	return ds 
 

def calculate_metrics (ds, net):
    #calculate rMSE and MAPE
	diff_list = []
	atft_list = []
	for sample, target in ds.getSequenceIterator(0):
		squared_diff = np.square((net.activate(sample) * active_max - target * active_max))
		ape = abs((net.activate(sample) - target)/target)
		diff_list.append(squared_diff)
		atft_list.append(ape)
          
	mse = float(sum (diff_list)) / float(len (diff_list))
	mape = float(sum(atft_list)) / float(len (atft_list)) 
 	rmse = math.sqrt(mse)
 	print ("The RMSE is ", rmse)
 	print ("The MAPE is ", mape)

def normalize (data):
	maxnum = max (data)
	normData = data / maxnum
	return (normData, maxnum)


def save_model (net):
	filename = "model.pkl"
	fileObject = open(filename, 'w')
	pickle.dump(net, fileObject)
	fileObject.close()

if __name__ == "__main__":
	# Load data 
	#filedir = "./data/consumption_data 2008.csv"
	filedir = "./data/consumption_data winter.csv"
	df = pd.read_csv(filedir)
	active = df["active"]
	#sub1 = df["Sub1"]
	date = df["DateTime"]
	li1 = active[0:4032].tolist()
	li2 = active[4032:].tolist()
	print(li2)    
	print(li2[1])
	ds = create_train_set (li1)  
	ds1 = create_train_set (li2)    

	print("Building network...")
	# Build a simple LSTM network with 1 input node, 5 LSTM cells and 1 output node
	net = buildNetwork(1, 5, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

	# Train the model
	print("Start training...")
	train_errors, EPOCHS, EPOCHS_PER_CYCLE = train (ds, net)

	print("Done training. Plotting error...")
	ploterror (train_errors, EPOCHS, EPOCHS_PER_CYCLE)

	print("Predicting price...")
	sam, tar = predict(ds1, net, date[4032:len(active)].tolist())

	print("Done predicting. Plotting prediction...")
	plot (sam, tar)
 
	print("Calculating mean squared error...")
	calculate_metrics (ds1, net)

	print("Saving model...")
	save_model(net)



