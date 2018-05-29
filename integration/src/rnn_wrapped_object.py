import tensorflow as tf
from postprocessing_functions import *
# import datetime
	
class trained_rnn_object:
	def __init__(self, max_in, min_in, max_out, min_out, model_path, batch_size, sess, tensor_name_dict = \
		{'x': "Reshape:0", 'y': "Reshape_1:0", 'T': "Reshape_2:0", 'outputs': "transpose_2:0", 'batch_size': "Placeholder_1:0", \
		'outp_keep': "outp_keep:0", 'states': ['rnn/while/Exit_2:0', 'rnn/while/Exit_3:0'], \
		'init_states': ['MultiRNNCellZeroState/GRUCellZeroState/zeros:0', 'MultiRNNCellZeroState/GRUCellZeroState_1/zeros:0']}, \
		mode = 'map', scale=None, Ts=0.032, input_dim=6, output_dim=3, rep_dist=400.0):
		# Caution this framework is for research purposes only and consists of a robotic end-effector simulation
		# The application of this framework on a real robot can be dangerous
		# For safe operation, additional safety requirements need to be fulfilled
		self.max_in = max_in # maximum unscaled input
		self.min_in = min_in # minimum unscaled input
		self.max_out = max_out # maximum unscaled output
		self.min_out = min_out # minimum unscaled output
		self.scale = scale # instead of a min-max-mapping we can also scale inputs
		self.mode = mode # needs to be 'map' or 'scale'

		self.sess = sess # Tensorflow session, needs to be passed
		self.model_path = model_path # Path of the trained model
		self.saver = tf.train.import_meta_graph(model_path + ".ckpt-40.meta") # Restore meta graph
		self.saver.restore(self.sess, model_path + ".ckpt-40") # Restore checkpoint
		self.TfGraph = tf.get_default_graph() # Get TF graph
		
		# Define tensors from graph for simulation (needed, as the original implementation may not be available)
		self.feedDict = {}
		self.x = self.TfGraph.get_tensor_by_name(tensor_name_dict['x']) # 
		self.y = self.TfGraph.get_tensor_by_name(tensor_name_dict['y']) #
		self.outputs = self.TfGraph.get_tensor_by_name(tensor_name_dict['outputs'])
		self.state = tuple([self.TfGraph.get_tensor_by_name(s_n) for s_n in tensor_name_dict['states']])
		self.initial_state = tuple([self.TfGraph.get_tensor_by_name(i_s) for i_s in tensor_name_dict['init_states']])
		self.bs = batch_size
		if 'T' in tensor_name_dict.keys():
			self.T = self.TfGraph.get_tensor_by_name(tensor_name_dict['T'])
			self.feedDict[self.T] = [1]
		if 'batch_size' in tensor_name_dict.keys():
			self.batch_size = self.TfGraph.get_tensor_by_name(tensor_name_dict['batch_size'])
			self.feedDict[self.batch_size] = batch_size
		if 'outp_keep' in tensor_name_dict.keys():
			self.outp_keep= self.TfGraph.get_tensor_by_name(tensor_name_dict['outp_keep'])
			self.feedDict[self.outp_keep] = 1.0 # artifact: we didn't use drop-out
		
		# to be passed:
		self.Ts = Ts # Sampling time (original work: 0.032 [s], caution: changing this value may change RNN performance)
		self.acc_sequence = None # Sequence of end effector acceleration output values
		self.vel_sequence = None # Sequence of end effector acceleration based velocity values
		self.loc_sequence = None # Sequence of end effector velocity based position values
		self.state_val = None # Stores past RNN state value
		self.output_dim = output_dim # output dimension (original work: 3)
		self.input_dim = input_dim # input dimension (original work: 6)
		self.rep_dist = rep_dist # Distance for the input feature that is based on repulsive forces, (original work: 400.0 [mm])

		# Workaround to run the model once, in the beginning (apparently harmonizes the runtime of the first two iterations in the real run with the rest.)
		self.feedDict[self.x] = np.repeat(np.ones(self.input_dim).reshape(1,1,self.input_dim), self.bs, axis=0)
		self.feedDict[self.y] = np.repeat(np.ones(self.output_dim).reshape(1,1,self.output_dim), self.bs, axis=0)
		# test_array = np.repeat(np.ones(self.input_dim).reshape(1,1,self.input_dim), 1, axis=0)
		# print test_array.shape
		_ , i_stat = self.sess.run([self.outputs, self.state], feed_dict=self.feedDict)
		self.feedDict[self.x] = np.repeat(np.ones(self.input_dim).reshape(1,1,self.input_dim), self.bs, axis=0)
		self.feedDict[self.y] = np.repeat(np.ones(self.output_dim).reshape(1,1,self.output_dim), self.bs, axis=0)
		self.feedDict[self.initial_state] = i_stat
		_ , __ = self.sess.run([self.outputs, self.state], feed_dict=self.feedDict)
		del self.feedDict[self.initial_state]
		# print len(self.feedDict.keys())

		print "\n \n You are advised to use inputs only in the range of: "
		print str(np.squeeze(self.max_in)) + "\n to \n" + str(np.squeeze(self.min_in))
		print "\n \n"

	def step(self, interactPos, goalPos, y = None, wristPos = np.array([np.nan]), idx=-1, numerical_integration = True, test_exception=True):
		#	interactPos:			Position of the other human's wrist
		#	goalPos:				Goal position for the RNN controlled end effector
		#	y:						Target acceleration values (redundant during this testing) which is used for masking, used 
		#							if: outputs = tf.cast(tf.cast(y, dtype=tf.bool), dtype=tf.float64)*outputs
		#	wristPos:				Position of controlled end effector. If the built-in integrator is used as only simulator, it solely requires
		#							values for the first 2 or 3 time steps
		# 	idx:					index of outer loop (must be 0 for the first time step) and then larger than 0
		#	numerical_integration:	Boolean - if the built-in numerical integrator is used - default
		#	test_exception:			Boolean - The numerical integration can be performed in different ways. In order to copy the one from the original 
		#							implementation set to True.

		if y == None:
			# c.f. outputs = outputs = tf.cast(tf.cast(y, dtype=tf.bool), dtype=tf.float64)*outputs in original training script
			y = np.ones(self.output_dim).reshape(1,1,self.output_dim) # used to ensure boolean masking

		# Get mapped [-1,1] input values, for Mingpan's code just devide by 1000
		if np.any(np.isnan(wristPos)):
			in_val = self.pack_and_map_or_scale_inputs(self.loc_sequence[0,-1,:].reshape(1,1,self.output_dim), goalPos, interactPos)
			# in_val = map_inputs(pack_data(self.loc_sequence[0,-1,:].reshape(1,1,self.output_dim), goalPos, interactPos, rep_dist=self.rep_dist), self.max_in, self.min_in)
		else:
			in_val = self.pack_and_map_or_scale_inputs(wristPos.reshape(1,1,self.output_dim), goalPos, interactPos)
			# in_val = map_inputs(pack_data(wristPos.reshape(1,1,self.output_dim), goalPos, interactPos, rep_dist=self.rep_dist), self.max_in, self.min_in)

		# Compute raw outputs from raw inputs
		if self.state_val == None:
			# default initial state is 0
			self.feedDict[self.x] = np.repeat(in_val, self.bs, axis=0)
			self.feedDict[self.y] = np.repeat(y, self.bs, axis=0)
			acc, fin_state = self.sess.run([self.outputs, self.state], feed_dict=self.feedDict)
		else:
			self.feedDict[self.x] = np.repeat(in_val, self.bs, axis=0)
			self.feedDict[self.y] = np.repeat(y, self.bs, axis=0)
			self.feedDict[self.initial_state] = self.state_val
			acc, fin_state = self.sess.run([self.outputs, self.state], feed_dict=self.feedDict)
		self.state_val = fin_state

		# Get mapped real world outputs, for Mingpan's Code, double check -> possibly replace map_outputs by simple multiplication
		# batch_size - normalize
		# Make an exception and transform the output to (1,1,dim) np.array()

		# acc = map_outputs(acc, self.max_out, self.min_out)
		# print acc.shape # in my version gives (1,1,3)
		if self.bs > 1:
			acc = np.squeeze(acc[0,:]).reshape(1,1,self.output_dim)
		acc = self.map_or_scale_outputs(acc)
		
		# Now compute the integrated quantities velocity (vel) and position (pos)
		# numerical integration: Integral_of_x[t] = Integral_of_x[t-1] + 0.5*(x[t-1] + x[t])
		if numerical_integration:
			# Default use with numerical_integration = True
			if idx == 0:
				# print self.feedDict
				self.acc_sequence = acc
				self.vel_sequence = np.zeros([1,1,self.output_dim])
				self.loc_sequence = wristPos.reshape(1,1,self.output_dim)
			elif idx==1 and test_exception == True:
				# Special case to test this version against a previously iplemented numerical integration method
				self.acc_sequence = np.hstack((self.acc_sequence, acc))
				self.vel_sequence = np.hstack((self.vel_sequence, self.vel_sequence[0,-1,:].reshape(1,1,self.output_dim) + 0.5*self.Ts*(self.acc_sequence[0,-1,:].reshape(1,1,self.output_dim) + self.acc_sequence[0,-2,:].reshape(1,1,self.output_dim))))
				self.loc_sequence = np.hstack((self.loc_sequence, self.loc_sequence[0,-1,:].reshape(1,1,self.output_dim)))
			else:
				self.acc_sequence = np.hstack((self.acc_sequence, acc))
				self.vel_sequence = np.hstack((self.vel_sequence, self.vel_sequence[0,-1,:].reshape(1,1,self.output_dim) + 0.5*self.Ts*(self.acc_sequence[0,-1,:].reshape(1,1,self.output_dim) + self.acc_sequence[0,-2,:].reshape(1,1,self.output_dim))))
				self.loc_sequence = np.hstack((self.loc_sequence, self.loc_sequence[0,-1,:].reshape(1,1,self.output_dim) + 0.5*self.Ts*(self.vel_sequence[0,-1,:].reshape(1,1,self.output_dim) + self.vel_sequence[0,-2,:].reshape(1,1,self.output_dim))))
		else:
			raise NotImplementedError
            
		return np.squeeze(self.acc_sequence[0,-1,:]), np.squeeze(self.vel_sequence[0,-1,:]), np.squeeze(self.loc_sequence[0,-1,:])

	def reset(self):
		self.acc_sequence = None
		self.vel_sequence = None
		self.loc_sequence = None
		self.state_val = None
		return True

	def get_sequences(self):
		return self.acc_sequence, self.vel_sequence, self.loc_sequence

	def map_or_scale_outputs(self, acc):
		if self.mode == 'map':
			return map_outputs(acc, self.max_out, self.min_out)
		elif self.mode == 'scale':
			return acc*self.scale
		else:
			raise NotImplementedError

	def pack_and_map_or_scale_inputs(self, wrist, goal, obstacle):
		if self.mode == 'map':
			return map_inputs(pack_data(wrist, goal, obstacle, rep_dist=self.rep_dist), self.max_in, self.min_in)
		elif self.mode =='scale':
			return pack_data(wrist/self.scale, goal/self.scale, obstacle/self.scale, rep_dist=self.rep_dist/self.scale)
		else:
			raise NotImplementedError