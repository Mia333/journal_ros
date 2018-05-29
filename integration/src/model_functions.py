# References:
# [1] https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# [2] https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
# [3] https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
# [4] https://www.tensorflow.org/tutorials/recurrent
# [5] https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# [6] https://stackoverflow.com/questions/42520418/how-to-multiply-list-of-tensors-by-single-tensor-on-tensorflow
# [7] https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# [8] https://github.com/aymericdamien/TensorFlow-Examples/issues/14
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


import numpy as np
import tensorflow as tf
from postprocessing_functions import *
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl



# This file contains model related functions

def rnnCell(no_units, f_bias, reusePar, type="Basic", act_fcn = None):
    # Builds an instance of an RNN Cell according to the specifications
    # no_units:         Number of neurons in each cell/layer
    # f_bias:           Forget bias value
    # reusePar:         Tensorflow reuse parameter
    # type:             Cell type
    if type == "Basic":
        return tf.contrib.rnn.BasicLSTMCell(no_units, forget_bias = f_bias, state_is_tuple=True, reuse=False)
    elif type == "GRU":
        if act_fcn == None:
            # In case no customized activation function is used for GRU (default: tanh)
            return tf.contrib.rnn.GRUCell(no_units, reuse=False)
        elif act_fcn == "triple_tanh":
            # In case a customized activation function is used for GRU
            return tf.contrib.rnn.GRUCell(no_units, activation = triple_tanh, reuse=False)
        else:
            raise ValueError('Unknown activation function')
    elif type == "SimpleRNN":
        return tf.contrib.rnn.BasicRNNCell(no_units, reuse=False)
    elif type == "UGRNN":
        if act_fcn == None:
            return tf.contrib.rnn.UGRNNCell(no_units, reuse=False)
        elif act_fcn == "triple_tanh":
            # In case a customized activation function is used for GRU
            return tf.contrib.rnn.UGRNNCell(no_units, activation = triple_tanh, reuse=False)
        else:
            raise ValueError('Unknown activation function')
    elif type == "LSTM":
        return tf.contrib.rnn.LSTMCell(no_units, use_peepholes = True, reuse=False)
    elif type == "BasicMod":
        return BasicLSTMCellMod(num_units=no_units, forget_bias = f_bias, state_is_tuple=True, reuse=False)
    elif type == "BasicMod2":
        return BasicLSTMCellMod2(num_units=no_units, forget_bias = f_bias, state_is_tuple=True, reuse=False)
    else:
        return tf.contrib.rnn.BasicLSTMCell(no_units, forget_bias = f_bias, state_is_tuple=True, reuse=False)

def cost_function(prediction, targets, time_steps):
    # Implements a cost function with mean squared error (MSE)
    # prediction:       Predicted tensor
    # targets:          Original tensor
    # time_steps:       Number of relevant time steps in tensor
    c = tf.reduce_sum(tf.multiply(prediction-targets, prediction-targets))/\
    tf.cast(tf.reduce_sum(time_steps), dtype=tf.float64)
    tf.Print(c, [c, "Hello"])
    return c

def numericalIntegration(tensor_series, time_steps, batch_size, dimensions, mask, delta_T):
    # This function implements numerical integration with the trapezoidal rule
    # tensor_series:    sequence(s) to integrate
    # time_steps:       Maximum number of time steps in batch/sequences
    # batch_size:       Number of sequences, batch size
    # dimensions:       Dimensionality of each single sequence
    # mask:             Indicates which values are no longe parte of the integration
    init_step = tensor_series[:,0,:] + 0.5*(tensor_series[:,1,:]-tensor_series[:,0,:])
    integral = tf.reshape(init_step, [batch_size,1,dimensions])
    for idx in xrange(1,time_steps-1,1):
        new_t_step = tf.reshape(integral[:,idx-1,:] + 0.5*(tensor_series[:,idx+1,:]-tensor_series[:,idx,:]), [batch_size,1,dimensions])
        integral = tf.concat( [integral, new_t_step], axis=1 )
    integral = integral*delta_T
    integral = tf.multiply(integral, mask)
    return integral

def generate_model_name(opt_method, lay_list, unit_list, data_set="rep_rot"):
    # This function generates a (unique) model name, e.g. for saving checkpoints
    # opt_method:       Optimizer method that is used
    # lay_list:         List of the layer configuration (c.f. config file)
    # unit_list:        List of number of units/neurons per layer (c.f. config file)
    # data_set:         Suffix describing a data_set
    mod_name = opt_method
    for idx in range(len(lay_list)):
        mod_name = mod_name + "_" + lay_list[idx] + str(unit_list[idx])
    mod_name = mod_name + "_" + data_set
    return mod_name

def triple_tanh(x):
    # This is a modified activation function that addresses a three-phase approach:
    #
    # x:                The input tensor
    #
    # Math: modified_activation = 
    # (tanh(scaling*(x-offset_1)) + tanh(scaling*(x-offset_2)) + tanh(scaling*(x-offset_3)))/3
    offset_1 = -2.0
    offset_2 = 0.0
    offset_3 = -offset_1
    scaling = 4.0
    modified_activation = tf.divide( tf.tanh( tf.scalar_mul(scaling, x - offset_1) ) + \
        tf.tanh( tf.scalar_mul(scaling, x - offset_2) ) + \
        tf.tanh( tf.scalar_mul(scaling, x - offset_3) ), 3.0 )
    return modified_activation

def scipy_opt_loss_callback(loss_tensor, config):
    file_n = "Comparison/" +  config  + "_loss_values_scipy.csv"
    with open(file_n, "a") as myfile:
        myfile.write(str(loss_tensor) + "\n")

class BasicLSTMCellMod(rnn_cell_impl.RNNCell):
    # Based on BasicLSTMCell 
    # https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
        num_units: int, The number of units in the LSTM cell.
        forget_bias: float, The bias added to forget gates (see above).
        state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
        activation: Activation function of the inner states.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
        """
        super(BasicLSTMCellMod, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be " \
                "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM)."""
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        concat = rnn_cell_impl._linear([inputs, h], 4 * self._num_units, True )

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = ( \
            tf.tanh(tf.scalar_mul(20.0, sigmoid(i) * self._activation(j) - tf.scalar_mul(0.5, c) )) * c * sigmoid(f + self._forget_bias) \
            + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)
        # new_c = tf.Print(new_c, [new_c, new_h])


        if self._state_is_tuple:
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

class BasicLSTMCellMod2(rnn_cell_impl.RNNCell):
    # Based on BasicLSTMCell 
    # https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
        num_units: int, The number of units in the LSTM cell.
        forget_bias: float, The bias added to forget gates (see above).
        state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
        activation: Activation function of the inner states.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
        """
        super(BasicLSTMCellMod2, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be " \
                "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM)."""
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        concat = rnn_cell_impl._linear([inputs, h], 4 * self._num_units, True )

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = ( \
            tf.tanh(tf.scalar_mul(20.0, tf.multiply( \
                (sigmoid(i) * self._activation(j) - tf.scalar_mul(0.5, c)), \
                (sigmoid(i) * self._activation(j) - tf.scalar_mul(0.5, c)) ))) \
            * c * sigmoid(f + self._forget_bias) \
            + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)
        # new_c = tf.Print(new_c, [new_c, new_h])


        if self._state_is_tuple:
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

