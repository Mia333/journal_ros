# Additional python functions:
# Import from custom files
from data_functions import *
from model_functions import *
from postprocessing_functions import *

# Import from shipped libraries
import scipy.io as sio
import numpy as np
import tensorflow as tf
import inspect
# from tensorflow.contrib import rnn
import csv
import matplotlib.pyplot as plt
import socket
# from tensorflow.python import debug as tf_debug
import sys
import ConfigParser
import os
from itertools import chain # https://stackoverflow.com/questions/14099872/concatenating-two-range-function-results
from itertools import product # https://stackoverflow.com/questions/18648626/for-loop-with-two-variables

import pandas as pd

def configuration_loader(sys_argv):
	# Load configuration specific data c.f. [9] 
	if len(sys_argv) > 1:
	    config_file = sys_argv[1]
	    # config_id = tf.constant(config_file[0:-4], dtype=tf.string)
	else:
	    config_file = "config_9.ini"
	    # config_file = "default_config.ini" ##### from Ben
	    # config_id = tf.constant("config_0", dtype=tf.string)
	config_path = 'Configuration_files/' + config_file
	config = ConfigParser.SafeConfigParser()
	if not os.path.exists(config_path):
	    config.read("Configuration_files/config_9.ini") 
	    # config.read("Configuration_files/default_config.ini") ##### from Ben
	    print "Configuration does not exist. Default configuration loaded"
	else:
	    config.read(config_path)
	    print "Configuration under " + config_path + " was loaded."
	# Training parameters
	file_name = "Data/" + config.get('Training_parameters', 'file_name')
	decay = config.getfloat('Training_parameters', 'decay')
	initial_learning_rate = config.getfloat('Training_parameters', 'initial_learning_rate')
	batches_per_epoch = config.getint('Training_parameters', 'batches_per_epoch')
	num_epochs = config.getint('Training_parameters', 'num_epochs')
	decay_start_epoch = config.getint('Training_parameters', 'decay_start_epoch')
	bs =  config.getint('Training_parameters', 'bs')
	num_samples = config.getint('Training_parameters', 'num_samples')
	load_results = config.getboolean('Training_parameters', 'load_results')
	is_training = config.getboolean('Training_parameters', 'is_training')
	load_existing_graph = config.getboolean('Training_parameters', 'load_existing_graph')
	optimizer_method = config.get('Training_parameters', 'optimizer_method')
	# Optional training parameters
	if config.has_option('Training_parameters', 'regularize_weights'):
	    regularize_weights = config.getboolean('Training_parameters', 'regularize_weights')
	else:
	    regularize_weights = False

	if config.has_option('Training_parameters', 'regularize_w_const'):
	    regularize_w_const = config.getfloat('Training_parameters', 'regularize_w_const')
	else:
	    regularize_w_const = 0.001

	if config.has_option('Training_parameters', 'regularize_acc'):
	    regularize_acc = config.getboolean('Training_parameters', 'regularize_acc')
	else:
	    regularize_acc = False

	if config.has_option('Training_parameters', 'regularize_a_const'):
	    regularize_a_const = config.getfloat('Training_parameters', 'regularize_a_const')
	else:
	    regularize_a_const = 0.001

	# Network parameters
	num_layers = config.getint('Network_parameters', 'num_layers')
	target_dimensions = config.getint('Network_parameters', 'target_dimensions')
	input_dimensions = config.getint('Network_parameters', 'input_dimensions')
	max_ts = config.getint('Network_parameters', 'max_ts')
	layer_list = config.get('Network_parameters', 'layer_list').split(",")
	neuron_list = map(int, config.get('Network_parameters', 'neuron_list').split(","))
	delta_t = config.getfloat('Network_parameters', 'delta_t')
	# Optional network parameters
	if config.has_option('Network_parameters', 'activation_fcn'):
	    activation_fcn = config.get('Network_parameters', 'activation_fcn')
	else:
	    activation_fcn = None
	if config.has_option('Network_parameters', 'rep_dist'):
	    rep_dist = config.getfloat('Network_parameters', 'rep_dist')
	else:
	    rep_dist = 150.0

	# Test parameters
	plotting = config.getboolean('Test_parameters', 'plotting')
	num_ts_simulation = config.getint('Test_parameters', 'num_ts_simulation')
	test_files = config.get('Test_parameters', 'test_files').split(",")
	save_test_files = config.get('Test_parameters', 'save_test_files').split(",")
	forward_indices = map(int, config.get('Test_parameters', 'forward_indices').split(","))
	backward_indices = map(int, config.get('Test_parameters', 'backward_indices').split(","))
	reaching_threshold = config.getint('Test_parameters', 'reaching_threshold')

	# Optional test parameters 
	if config.has_option('Test_parameters', 'add_noise'):
	    add_noise = config.getboolean('Test_parameters', 'add_noise')
	else:
	    add_noise = False
	# Optional test parameters 
	if config.has_option('Test_parameters', 'static_simulation'):
	    static_simulation = config.getboolean('Test_parameters', 'static_simulation')
	else:
	    static_simulation = False

	return config_file, config_path, config, file_name, decay, initial_learning_rate, batches_per_epoch, num_epochs, decay_start_epoch, bs, num_samples, \
		load_results, is_training, load_existing_graph, optimizer_method, regularize_weights, regularize_w_const, regularize_acc, regularize_a_const, \
		num_layers, target_dimensions, input_dimensions, max_ts, layer_list, neuron_list, delta_t, activation_fcn, rep_dist, plotting, num_ts_simulation, \
		test_files, save_test_files, forward_indices, backward_indices, reaching_threshold, add_noise, static_simulation

def load_information_for_rnn_tests(config_name):
	config_file, config_path, config, file_name, decay, initial_learning_rate, batches_per_epoch, num_epochs, decay_start_epoch, bs, num_samples, \
	load_results, is_training, load_existing_graph, optimizer_method, regularize_weights, regularize_w_const, regularize_acc, regularize_a_const, \
	num_layers, target_dimensions, input_dimensions, max_ts, layer_list, neuron_list, delta_t, activation_fcn, rep_dist, plotting, num_ts_simulation, \
	test_files, save_test_files, forward_indices, backward_indices, reaching_threshold, add_noise, static_simulation = configuration_loader(config_name)

	model_name = generate_model_name(optimizer_method, layer_list, neuron_list) + file_name[5:-4]
	if int(config_file[7:-4]) > 59:
		if len(neuron_list) == 2 and layer_list[0] == "Basic" and layer_list[1] == "Basic":
			# These dicts need to be passed to "tensor_name_dict" in rnn_wrapped_object.py 
			# for Ben Pfirrmann's 2 layer architectures (WITH zero mask i.e. config 60 ...)
			# For LSTM
			tensor_name_dict = \
			{'x': "Reshape:0", 'y': "Reshape_1:0", 'T': "Reshape_2:0", 'out': "mul_6:0", 'batch_size': "Placeholder_1:0", \
			'outp_keep': "outp_keep:0", 'states': ['rnn/while/Exit_2:0', 'rnn/while/Exit_3:0', 'rnn/while/Exit_4:0', 'rnn/while/Exit_5:0'], \
			'init_states': ['MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros:0', 'MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1:0', \
			'MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros:0', 'MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1:0']}
		elif len(neuron_list) == 2 and layer_list[0] == "GRU" and layer_list[1] == "GRU":
			# For GRU and GRU with triple-tanh
			tensor_name_dict = \
			{'x': "Reshape:0", 'y': "Reshape_1:0", 'T': "Reshape_2:0", 'out': "mul_6:0", 'batch_size': "Placeholder_1:0", \
			'outp_keep': "outp_keep:0", 'states': ['rnn/while/Exit_2:0', 'rnn/while/Exit_3:0'], \
			'init_states': ['MultiRNNCellZeroState/GRUCellZeroState/zeros:0', 'MultiRNNCellZeroState/GRUCellZeroState_1/zeros:0']}
		else:
			raise ValueError('No tensor_name_dict saved')
		saver_path = "New_Saves/" + model_name
	else:
		if len(neuron_list) == 2 and layer_list[0] == "Basic" and layer_list[1] == "Basic":
			# These dicts need to be passed to "tensor_name_dict" in rnn_wrapped_object.py 
			# for Ben Pfirrmann's 2 layer architectures (WITHOUT zero mask i.e. config 5 ... config 50)
			# For LSTM
			tensor_name_dict = \
			{'x': "Reshape:0", 'y': "Reshape_1:0", 'T': "Reshape_2:0", 'out': "transpose_2:0", 'batch_size': "Placeholder_1:0", \
			'outp_keep': "outp_keep:0", 'states': ['rnn/while/Exit_2:0', 'rnn/while/Exit_3:0', 'rnn/while/Exit_4:0', 'rnn/while/Exit_5:0'], \
			'init_states': ['MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros:0', 'MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1:0', \
			'MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros:0', 'MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1:0']}
		elif len(neuron_list) == 2 and layer_list[0] == "GRU" and layer_list[1] == "GRU":
			# For GRU and GRU with triple-tanh
			tensor_name_dict = \
			{'x': "Reshape:0", 'y': "Reshape_1:0", 'T': "Reshape_2:0", 'out': "transpose_2:0", 'batch_size': "Placeholder_1:0", \
			'outp_keep': "outp_keep:0", 'states': ['rnn/while/Exit_2:0', 'rnn/while/Exit_3:0'], \
			'init_states': ['MultiRNNCellZeroState/GRUCellZeroState/zeros:0', 'MultiRNNCellZeroState/GRUCellZeroState_1/zeros:0']}
		else:
			raise ValueError('No tensor_name_dict saved')
		saver_path = "Saves/" + model_name
	if activation_fcn in ["triple_tanh"]:
	    saver_path = saver_path + "_mod_act"

	no_sam = len(forward_indices) + len(backward_indices)

	# test_files is only used with single test file and no longer a list
	iD, tD, lD, mx_in, mn_in, mx_out, mn_out, wrist, obstacle, goal = loadData("Data/"+test_files[0], "test", no_sam)

	return config_file, config_path, config, file_name, decay, initial_learning_rate, batches_per_epoch, num_epochs, decay_start_epoch, bs, num_samples, \
	load_results, is_training, load_existing_graph, optimizer_method, regularize_weights, regularize_w_const, regularize_acc, regularize_a_const, \
	num_layers, target_dimensions, input_dimensions, max_ts, layer_list, neuron_list, delta_t, activation_fcn, rep_dist, plotting, num_ts_simulation, \
	test_files, save_test_files, forward_indices, backward_indices, reaching_threshold, add_noise, static_simulation, saver_path, no_sam, iD, tD, lD, \
	mx_in, mn_in, mx_out, mn_out, wrist, obstacle, goal, tensor_name_dict

print configuration_loader(sys.argv)


