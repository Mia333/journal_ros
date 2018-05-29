# Import from custom files
from data_functions import *
from model_functions import *
from postprocessing_functions import *
from beta_new_modular_functions import *
from rnn_wrapped_object import *

# Import from shipped libraries
import scipy.io as sio
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
# import statistics


import datetime
import time

import sys
import ConfigParser
import os

with tf.Session() as sess:
	config_file, config_path, config, file_name, decay, initial_learning_rate, batches_per_epoch, num_epochs, \
	decay_start_epoch, bs, num_samples, load_results, is_training, load_existing_graph, optimizer_method, \
	regularize_weights, regularize_w_const, regularize_acc, regularize_a_const, num_layers, target_dimensions, \
	input_dimensions, max_ts, layer_list, neuron_list, delta_t, activation_fcn, rep_dist, plotting, num_ts_simulation, \
	test_files, save_test_files, forward_indices, backward_indices, reaching_threshold, add_noise, static_simulation, \
	saver_path, no_sam, iD, tD, lD, mx_in, mn_in, mx_out, mn_out, wrist, obstacle, goal, tensor_name_dict \
	= load_information_for_rnn_tests(sys.argv)

	# Here we initialize a new object of the class "trained_rnn_object" with all associated value
	#new_rnn = trained_rnn_object(max_in = mx_in, min_in = mn_in, max_out = mx_out, min_out = mn_out, model_path = saver_path + "/my_save_file", batch_size=1, sess=sess, tensor_name_dict = tensor_name_dict, \
	#	mode = 'map', scale = None, Ts=delta_t, input_dim=input_dimensions, output_dim=target_dimensions, rep_dist=rep_dist)
	
	
	# Redundant: old implementation
	# new_rnn = trained_rnn_object(mx_in, mn_in, mx_out, mn_out, saver_path + "/my_save_file", sess, tensor_name_dict = tensor_name_dict, \
	#	Ts=delta_t, input_dim=input_dimensions, output_dim=target_dimensions, rep_dist=rep_dist)
	
	
	# For Mingpan's code, something like this (load_inforation_for_rnn_tests can be used to load the test set from a specific config as well)
	new_rnn = trained_rnn_object(max_in = None, min_in = None, max_out = None, min_out = None, model_path = "checkpoints/rf20_16_fix40/model", batch_size=10, sess=sess, \
		tensor_name_dict = {'x': "Placeholder:0", 'y': "Placeholder_1:0", 'outputs': "add:0", \
		'states': ['rnn/while/Exit_2:0', 'rnn/while/Exit_3:0', 'rnn/while/Exit_4:0', 'rnn/while/Exit_5:0'], \
		'init_states': ['MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros:0', 'MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1:0', \
		'MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros:0', 'MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1:0']}, \
		mode = 'scale', scale = 1000.0, Ts=0.032, input_dim=6, output_dim=3, rep_dist=400.0)

	for idx in range(wrist.shape[0]):
		# Loop over all test trajectories
		wrist_ = np.expand_dims(np.transpose(wrist[idx], (1,0)), axis=0)
		obstacle_ = np.expand_dims(np.transpose(obstacle[idx], (1,0)) , axis=0)
		goal_ = np.expand_dims(np.transpose(goal[idx], (1,0)), axis=0)
		time_deltas = []
		print 'wrist_shape: ', wrist[1].shape
		print 'wrist_ _shape: ', wrist_.shape
		print 'obstacle_ _shape: ', obstacle_.shape
		print 'goal_ _shape: ', goal_.shape

		for idx_2 in range(num_ts_simulation):
			# Loop over all time steps
			time_0 = datetime.datetime.now() # command to monitor runtime

			# Case distinction: 
			if idx_2 < 3: # initialize wrist position for first three time steps (in real simulation input should be given for full time series)
				g = goal_[0,0,:]
				o = obstacle_[0,idx_2,:]
				w = wrist_[0,idx_2,:]
			elif idx_2 > 2 and idx_2 < obstacle_.shape[1]: # need dynamic value here
				g = goal_[0,idx_2,:]
				o = obstacle_[0,idx_2,:]
				w = np.empty(list(g.shape))
				w.fill(np.nan)
			else:
				g = goal_[0,-1,:]
				o = obstacle_[0,-1,:]
				w = np.empty(list(g.shape))
				w.fill(np.nan)
			time_1 = datetime.datetime.now()
			print('o.shape: ',o.shape)
			new_rnn.step(o, g, y = None, wristPos = w, idx=idx_2, numerical_integration = True, test_exception=True)
			time_2 = datetime.datetime.now()
			time_deltas.append( (time_2 - time_1).microseconds/1000.0 + (time_2 - time_1).seconds*1000.0)
		accs, vels, locs = new_rnn.get_sequences()
		new_rnn.reset() # Reset RNN acceleration, velocity, location, state

		# evaluate runtime metrics
		print "Trajectory " + str(idx) + "\tRun time step-wise statitistics in ms \tMean: " + str(np.mean(time_deltas)) + "\t Median: " + str(np.median(time_deltas)) + "\t Std.-dev: " \
		+  str(np.std(time_deltas)) + "\t Max: " + str(np.max(time_deltas)) + "\t Min: " + str(np.min(time_deltas))

		# Plot resulting trajectories in 2D projection
		fig = plt.figure()
		splot = fig.add_subplot(111)
		hndl1, = splot.plot(locs[0,:,1], locs[0,:,0], label='sim loc' )
		hndl2, = splot.plot(wrist_[0,:,1], wrist_[0,:,0], label='orig wrist')
		hndl3, = splot.plot(obstacle_[0,:,1], obstacle_[0,:,0], label='2nd wrist')
		splot.legend(handles=[hndl1, hndl2, hndl3])
		plt.show()

		"""
		# Plot histogram for single rnn-step runtime
		plt.hist(time_deltas)
		plt.title("Histogram: single run time values")
		plt.xlabel("Run time [ms]")
		plt.ylabel("Frequency")
		fig = plt.gcf()

		# Plot time series of single rnn-step runtime
		fig = plt.figure()
		splot = fig.add_subplot(111)
		hndl1, = splot.plot(time_deltas, label='Run time RNN-step')
		hndl2, = splot.plot([32 for i in range(200)], label='Sampling time: 32 ms')
		splot.legend(handles=[hndl1, hndl2])
		fig.suptitle("Runtime for single RNN steps vs. sampling time")
		splot.set_xlabel('Steps')
		splot.set_ylabel('Runtime [ms]')
		plt.show()
		"""
	sess.close()