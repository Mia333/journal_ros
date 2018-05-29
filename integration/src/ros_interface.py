########################################
#            ROS Interface             #
########################################

# Additional python functions:
# Import from custom files
from data_functions import *
from model_functions import *
from postprocessing_functions import *
from beta_new_modular_functions import *
from rnn_wrapped_object import *
from rnn_wrapped_object import trained_rnn_object


# Import from shipped libraries
import scipy.io as sio
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

# import ros libraries
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose

# import statistics


import datetime
import time

import sys
import ConfigParser
import os

#Class
class rosinterface(object):

	def __init__(self, sess, goal_pos = np.zeros(3)):
		self.wrist_pos = None
		self.obstacle_pos = None
		self.goal_pos =  goal_pos  # Define goal_ position !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
		self.new_pos = None
		self.back_pos = None
	# a list contains real trajectory to publish
		self.traj = []

	# sequence from subscriber  
		self.acc_seq = None # Sequence of end effector acceleration output values
		self.vel_seq = None # Sequence of end effector acceleration based velocity values
		self.loc_seq = None # Sequence of end effector velocity based position values

		self.sess = sess  # Tensorflow session, needs to be passed

		self.idx = 0

		config_file, config_path, config, file_name, decay, initial_learning_rate, batches_per_epoch, num_epochs, \
		decay_start_epoch, bs, num_samples, load_results, is_training, load_existing_graph, optimizer_method, \
		regularize_weights, regularize_w_const, regularize_acc, regularize_a_const, num_layers, target_dimensions, \
		input_dimensions, max_ts, layer_list, neuron_list, delta_t, activation_fcn, rep_dist, plotting, num_ts_simulation, \
		test_files, save_test_files, forward_indices, backward_indices, reaching_threshold, add_noise, static_simulation, \
		saver_path, no_sam, iD, tD, lD, mx_in, mn_in, mx_out, mn_out, wrist, obstacle, goal, tensor_name_dict \
		= load_information_for_rnn_tests(['', 'config_9.ini']) # (sys.argv)


	# Here we initialize a new object of the class "trained_rnn_object" with all associated value
	#new_rnn = trained_rnn_object(max_in = mx_in, min_in = mn_in, max_out = mx_out, min_out = mn_out, model_path = saver_path + "/my_save_file", batch_size=1, sess=sess, tensor_name_dict = tensor_name_dict, \
	#	mode = 'map', scale = None, Ts=delta_t, input_dim=input_dimensions, output_dim=target_dimensions, rep_dist=rep_dist)
	
	
	# Redundant: old implementation
	# new_rnn = trained_rnn_object(mx_in, mn_in, mx_out, mn_out, saver_path + "/my_save_file", sess, tensor_name_dict = tensor_name_dict, \
	#	Ts=delta_t, input_dim=input_dimensions, output_dim=target_dimensions, rep_dist=rep_dist)


	# For Mingpan's code, something like this (load_inforation_for_rnn_tests can be used to load the test set from a specific config as well)

		self.new_rnn = trained_rnn_object(max_in = None, min_in = None, max_out = None, min_out = None, model_path = "checkpoints/rf20_16_fix40/model", batch_size=10, sess=sess, \
			tensor_name_dict = {'x': "Placeholder:0", 'y': "Placeholder_1:0", 'outputs': "add:0", \
			'states': ['rnn/while/Exit_2:0', 'rnn/while/Exit_3:0', 'rnn/while/Exit_4:0', 'rnn/while/Exit_5:0'], \
			'init_states': ['MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros:0', 'MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1:0', \
			'MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros:0', 'MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1:0']}, \
			mode = 'scale', scale = 1000.0, Ts=0.032, input_dim=6, output_dim=3, rep_dist=400.0)


	def callObstacle(self, obstacle_msg):
		#rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data) # not important
		self.obstacle_pos = obstacle_msg.pose.position

	def callWrist(self, sess, wrist_msg):
		#rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data) # not important 
		self.wrist_pos = wrist_msg.pose.position


		# Step func: calculate the new position for next time step

		g = self.goal_pos
		w = self.wrist_pos
		o = self.obstacle_pos

		assert (g.shape==(3,)) and (w.shape == (3,)) and (o.shape==(3,)), 'Wrong input shape'

		self.new_rnn.step(o, g, y = None, wristPos = w, idx=self.idx, numerical_integration = True, test_exception=True) ###### idx_2
		accs, vels, locs = self.new_rnn.get_sequences()  
		self.acc_seq = accs # Sequence of end effector acceleration output values
		self.vel_seq = vels # Sequence of end effector acceleration based velocity values
		self.loc_seq = locs
		self.idx += 1


	# Subscriber for NN
	def get_pos(self ):

		rospy.init_node('integration', anonymous=True) 

		rospy.Subscriber("qualisys/obstacle_pos", PoseStamped, callObstacle)
		rospy.Subscriber("qualisys/wrist_pos", PoseStamped, callWrist)     # msg from contrlloer

		# spin() simply keeps python from exiting until this node is stopped
		rospy.spin()


	# Publisher that publishes the new wrist_position 
	def set_pos(self):
		pub = rospy.Publisher('new_pos', PoseStamped, queue_size=10) #???? type, queue???????
		rospy.init_node('integration', anonymous=True)
		rate = rospy.Rate(30) # 30hz


		time_1 = datetime.datetime.now()
		time_2 = datetime.datetime.now()
		time_delta = (time_2 - time_1).microseconds/1000.0 + (time_2 - time_1).seconds*1000.0	

		while time_delta < 3000 and (not rospy.is_shutdown()):
			self.new_pos = self.loc_seq[0,-1,:]
			self.traj.append(self.new_pos)	
			rospy.loginfo("publisher new position at time: %s" % rospy.get_time())
			pos_msg = PoseStamped() # position , orientation
			pos_msg.pose.position = self.new_pos 
			pub.publish(pos_msg)
			rate.sleep()

			time_2 = datetime.datetime.now()
			time_delta = (time_2 - time_1).microseconds/1000.0 + (time_2 - time_1).seconds*1000.0	


		while (not len(self.traj) == 0 ) and (not rospy.is_shutdown()):
			self.back_pos = self.traj.pop() 
			rospy.loginfo("publisher new position at time: %s" % rospy.get_time())
			pos_msg = PoseStamped() # position , orientation
			pos_msg.pose.position = self.back_pos 
			pub.publish(pos_msg)
			rate.sleep()

# main func
#if __name__ == '__main__':

#	with tf.Session() as sess:
#		rosint = rosinterface(sess,  goal_pos = np.zeros(3))
#		rosint.get_pos()
#		try:
#			rosint.set_pos()
#		except rospy.ROSInterruptException:
#			pass


    #while Button:
	#	rosint = rosinterface()
	#	rosint.get_pos()
	#	try:
	#		rosint.set_pos()
	#	except rospy.ROSInterruptException:
   
    # how to stop it

	#listener()
