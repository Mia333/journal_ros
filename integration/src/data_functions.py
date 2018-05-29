# References:
# [1] https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# [2] https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
# [3] https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
# [4] https://www.tensorflow.org/tutorials/recurrent
# [5] https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# [6] https://stackoverflow.com/questions/42520418/how-to-multiply-list-of-tensors-by-single-tensor-on-tensorflow
# [7] https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# [8] https://github.com/aymericdamien/TensorFlow-Examples/issues/14

import scipy.io as sio
import numpy as np
import tensorflow as tf

# this file contains functions and classes for the data pipeline

class data_handler(object):
    # This is a class that defines an object for user controlled batch-fetching
    def __init__(self, filename, mode, no_samp, batchSize, noTrajectories, outerShellName = 'exportData'):
        # Call as constructor
        # filename:             file name of data to be loaded (for format see matlab)
        # mode:                 "test"  or "training"
        # no_samp:              Number of samples
        # batchSize:            Batch size in witch data is fetched
        # noTrajectories:       Number of single trajectories in the data set
        # outerShellName:       Specifies the variable/name in which the data was exported on the MATLAB side
        self.filename = filename
        self.mode = mode
        self.no_samp = no_samp;
        self.batchSize = batchSize
        self.noTrajectories = noTrajectories
        self.rawData = sio.loadmat(filename)
        self.data = self.rawData[outerShellName]
        self.outerShellName = outerShellName
        self.batchIdx = 0
        self.inputs = None
        self.targets = None
        self.lengths = None
        self.max_in = self.data[6,0]
        self.min_in = self.data[7,0]
        self.max_out = self.data[8,0]
        self.min_out = self.data[9,0]
        self.wrist = None
        self.obstacle = None
        self.goal = None

    def shuffle_and_extract_data(self):
        # This function shuffles the data set with a random permutation and sets the internal data accordingly
        self.batchIdx = 0
        rand_perm = np.random.permutation(self.no_samp)
        # print rand_perm
        self.inputs = self.data[0,:]
        if self.mode == "training":
            print "Inputs shuffled"
            self.inputs = self.inputs[rand_perm]
        self.targets = self.data[1,:]
        if self.mode == "training":
            print "Targets shuffled"
            self.targets = self.targets[rand_perm]
        self.lengths = self.data[2,:]
        if self.mode == "training":
            print "Lengths shuffled"
            self.lengths = self.lengths[rand_perm]
        if not self.mode == "training":
            self.wrist = self.data[3,:]
            self.obstacle = self.data[4,:]
            self.goal = self.data[5,:]
            
    def getBatch(self):
        # Extracts a batch in the form: [batchSize x noTimesteps x dimensionality]
        # circular indexing: if end reached, start from beginning
        startIdx = ((1 + self.batchIdx*self.batchSize) % self.noTrajectories) - 1
        endIdx = ((self.batchIdx + 1)*self.batchSize % self.noTrajectories) - 1          
        if endIdx > startIdx:
            if (endIdx == self.noTrajectories - 1):
                inputBatch = self.inputs[startIdx:]
                targetBatch = self.targets[startIdx:]
                timeSteps = self.lengths[startIdx:]

            else:
                inputBatch = self.inputs[startIdx:endIdx+1]
                targetBatch = self.targets[startIdx:endIdx+1]
                timeSteps = self.lengths[startIdx:endIdx+1]
        elif endIdx == startIdx and self.batchSize == 1:
            inputBatch = self.inputs[[startIdx]]
            targetBatch = self.targets[[startIdx]]
            timeSteps = self.lengths[[startIdx]]
        else:
            # This focusses on the case of reusing data in one epoch.
            # However, this case will typically not occur
            inputBatch = np.concatenate((self.inputs[startIdx:], self.inputs[0:endIdx+1]), axis=0)
            targetBatch = np.concatenate((self.targets[startIdx:], self.targets[0:endIdx+1]), axis = 0)
            timeSteps = np.concatenate((self.lengths[startIdx:], self.lengths[0:endIdx+1]), axis = 0)
        inputBatchList = inputBatch.tolist()
        targetBatchList = targetBatch.tolist()
        timeStepsList = timeSteps.tolist()
        inputBatchTf = np.transpose(np.stack(inputBatchList, axis=0), (0, 2, 1))
        targetBatchTf = np.transpose(np.stack(targetBatchList, axis=0), (0, 2, 1))
        timeStepsTf = np.squeeze(np.transpose(np.stack(timeStepsList, axis=0), (0, 2, 1)), axis = [1, 2])
        self.batchIdx = self.batchIdx + 1
        if self.batchIdx == 34:
            self.batchIdx = 0
        return inputBatchTf, targetBatchTf, timeStepsTf

def loadData(filename, mode, no_samp, outerShellName = 'exportData'):
    # filename:             file name of data to be loaded (for format see matlab)
    # mode:                 "test"  or "training"
    # no_samp:              Number of samples
    # outerShellName:       Specifies the variable/name in which the data was exported on the MATLAB side
    # Deprecated: this is currently used in the "run_and_save_simulation" function
    # It can/shall be replaced in a later version with the updated batching from data_handler or data_instance
    rand_perm = np.random.permutation(no_samp)
    rawData = sio.loadmat(filename)
    if not outerShellName == None:
        data = rawData[outerShellName]
    else:
        data = rawData
    inputs = data[0,:]
    if mode == "training":
        print "Inputs shuffled"
        inputs = inputs[rand_perm]
    targets = data[1,:]
    if mode == "training":
        print "Targets shuffled"
        targets = targets[rand_perm]
    lengths = data[2,:]
    if mode == "training":
        print "Lengths shuffled"
        lengths = lengths[rand_perm]
    max_in = data[6,0] # as the minimum is passed as a same value in all the columns
    min_in = data[7,0]
    max_out = data[8,0]
    min_out = data[9,0]
    if mode == "training":
        return inputs, targets, lengths, max_in, min_in, max_out, min_out
    else:
        wrist = data[3,:]
        obstacle = data[4,:]
        goal = data[5,:]
        return inputs, targets, lengths, max_in, min_in, max_out, min_out, wrist, obstacle, goal

def getBatch(batchIdx, batchSize, noTrajectories, inputData, targetData, lengthData):
    # Deprecated: this is currently used in the "run_and_save_simulation" function
    # It can/shall be replaced in a later version with the updated batching from data_handler or data_instance
    # Extracts a batch in the form: [batchSize x noTimesteps x dimensionality]
    # circular indexing: if end reached, start from beginning
    startIdx = ((1 + batchIdx*batchSize) % noTrajectories) - 1
    endIdx = ((batchIdx + 1)*batchSize % noTrajectories) - 1
    # print(" ")
    #print(startIdx, endIdx)               
    if endIdx > startIdx:
        if (endIdx == noTrajectories - 1):
            inputBatch = inputData[startIdx:]
            targetBatch = targetData[startIdx:]
            timeSteps = lengthData[startIdx:]

        else:
            inputBatch = inputData[startIdx:endIdx+1]
            targetBatch = targetData[startIdx:endIdx+1]
            timeSteps = lengthData[startIdx:endIdx+1]
    elif endIdx == startIdx and batchSize == 1:
        # print "Case 2"
        inputBatch = inputData[[startIdx]]
        targetBatch = targetData[[startIdx]]
        timeSteps = lengthData[[startIdx]]
        # print timeSteps
    else:
        # This focusses on the case of reusing data in one epoch.
        # However, this case will typically not occur
        inputBatch = np.concatenate((inputData[startIdx:], inputData[0:endIdx+1]), axis=0)
        targetBatch = np.concatenate((targetData[startIdx:], targetData[0:endIdx+1]), axis = 0)
        timeSteps = np.concatenate((lengthData[startIdx:], lengthData[0:endIdx+1]), axis = 0)
    inputBatchList = inputBatch.tolist()
    targetBatchList = targetBatch.tolist()
    timeStepsList = timeSteps.tolist()
    inputBatchTf = tf.convert_to_tensor(np.transpose(np.stack(inputBatchList, axis=0), (0, 2, 1)))
    targetBatchTf = tf.convert_to_tensor(np.transpose(np.stack(targetBatchList, axis=0), (0, 2, 1)))
    timeStepsTf = tf.squeeze(tf.convert_to_tensor(np.transpose(np.stack(timeStepsList, axis=0), (0, 2, 1))), axis = [1, 2])
    return inputBatchTf, targetBatchTf, timeStepsTf

def inspectData(inBatch, tarBatch, tSteps, inDim, tarDim, timeSteps, batchSize, batchIdx="NoIndexGiven"):
    # Deprecated
    # inBatch:              Input batch
    # tarBatch:             Target batch
    # tSteps:               Time Steps
    # inDim:                input dimension
    # tardime:              Target dimension
    # timeSteps:            Time steps
    # batchSize:            Batch Size
    # batchIdx:             Number/index of batch
    # A simple function to verify that the data is loaded correctly w.r.t the original data in matlab
    # More functions could be added if necessary
    def _simpleOutput(_bt, _dim, name):
        print ""
        print name + " no: " + str(batchIdx) + " is of type: " + str(type(tarBatch)) + " and size: " + str(tarBatch.shape)
        for d in range(0, _dim):
            print "from: " + np.array_str(_bt[0][0][d]) + ", " + np.array_str(_bt[1][0][d]) + "..." + np.array_str(_bt[batchSize-2][0][d] ) \
                + ", " + np.array_str(_bt[batchSize-1][0][d])   + "   to   " + np.array_str(_bt[0][timeSteps-2][d]) + ", " + np.array_str(_bt[1][timeSteps-2][d]) \
                + "..." + np.array_str(_bt[batchSize-2][timeSteps-2][d]) + ", " + np.array_str(_bt[batchSize-1][timeSteps-2][d])
        print "With time steps over batch: " + np.array_str(tSteps)
    _simpleOutput(inBatch, inDim, "Input batch")
    _simpleOutput(tarBatch, tarDim, "Target batch")

def data_instance(filename, no_samp, batchSize, noTrajectories, numBatches, max_t_steps, in_dim, out_dim, outerShellName = 'exportData'):
    # This data_instance function works similar as the class data_handler but implements a queue-runner
    # for cases where the training is automatically controlled by the session (e.g. ScipyOptimizerInterface)
    # filename:             file name of data to be loaded (for format see matlab)
    # no_samp:              Number of samples
    # batchSize:            Batch size in witch data is fetched
    # noTrajectories:       Number of single trajectories in the data set
    # outerShellName:       Specifies the variable/name in which the data was exported on the MATLAB side
    rawData = sio.loadmat(filename)
    data = rawData[outerShellName]
    in_raw = np.transpose(np.stack(data[0,:].tolist(), axis=0), (0,2,1) )
    tar_raw = np.transpose(np.stack(data[1,:].tolist(), axis=0), (0,2,1) )
    len_raw = np.squeeze(np.transpose(np.stack(data[2,:].tolist(), axis=0), (0,2,1) ), axis = [1, 2])
    i = tf.train.range_input_producer(numBatches, shuffle=True).dequeue()
    inputs_ = tf.convert_to_tensor(in_raw, dtype=tf.float64)
    targets_ = tf.convert_to_tensor(tar_raw, dtype=tf.float64)
    lengths_ = tf.convert_to_tensor(len_raw, dtype=tf.float64)
    indices = tf.convert_to_tensor(range(noTrajectories), dtype=tf.int32)
    def randperm_fcn(num_samp):
    	return np.random.permutation(num_samp)
    n_samp_t = tf.convert_to_tensor(no_samp, dtype=tf.int64)
    rand_perm_op = tf.py_func(randperm_fcn, [n_samp_t], tf.int64)
    # rand_perm_op = tf.convert_to_tensor(np.random.permutation(no_samp))
    rand_perm_op = tf.Print(rand_perm_op, ["Generated a random permutation for indexing: ", rand_perm_op[0:5]])
    # Old implementation with index-based batching
    with tf.control_dependencies([rand_perm_op]):
    	# inputs_ = tf.gather(inputs_, rand_perm_op)
    	# targets_ = tf.gather(targets_, rand_perm_op)
    	# lengths_ = tf.gather(lengths_, rand_perm_op)
    	# lengths_ = tf.Print(lengths_, ["inputs_, targes_, lengths_ were shuffled with a random permutation:", rand_perm_op[0:5]])
    	inputs = tf.strided_slice(inputs_, [i*batchSize, 0, 0], [(i+1)*batchSize, max_t_steps, in_dim])
    	targets = tf.strided_slice(targets_, [i*batchSize, 0, 0], [(i+1)*batchSize, max_t_steps, out_dim])
    	lengths = tf.strided_slice(lengths_, [i*batchSize], [(i+1)*batchSize])
	# in_list = tf.unstack(inputs_, num=None, axis=0)
	# tar_list = tf.unstack(targets_, num=None, axis=0)
	# len_list = tf.unstack(lengths_, num=None, axis=0)
    # New implementation
    # inputs, targets, lengths, ind_s = tf.train.shuffle_batch([in_list, tar_list, len_list, indices], batchSize, 5, 4, enqueue_many = True)
    inputs = tf.Print(inputs, ["Fetched input batch with shape: " + str(inputs.shape) + ", output batch with shape " + str(targets.shape) + " lengths batch with shape: " + str(lengths.shape)])
    return inputs, targets, lengths, i
