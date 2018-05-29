# References:
# [1] https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# [2] https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
# [3] https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
# [4] https://www.tensorflow.org/tutorials/recurrent
# [5] https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# [6] https://stackoverflow.com/questions/42520418/how-to-multiply-list-of-tensors-by-single-tensor-on-tensorflow
# [7] https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# [8] https://github.com/aymericdamien/TensorFlow-Examples/issues/14
# [9] https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib

import tensorflow as tensorflow
import numpy as np
from data_functions import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns; sns.set()
import matplotlib.cm as mplcm # [9]
import matplotlib.colors as colors # [9]
import copy
import csv


def pack_data(input_vals, goal_pos, obst_pos, feat_1 = 'H2T', feat_2='Rep', rep_dist=150.0):
    # Computes network inputs based on (actual/simulated observations):
    # input_vals:               end effector positions
    # goal_pos:                 Goal positions
    # obst_pos:                 Obstacle positions
    # feat_1:                   First feature (H2T for Hand-to-Target vector)
    # feat_2:                   Second feature (H2O for Hand-to-Obstacle vector or Rep for term inspired by potential fields)
    goal_pos = np.expand_dims(np.expand_dims(goal_pos, axis=0), axis=0)
    if feat_1 == "H2T" and feat_2 == "H2O":
        h2t = goal_pos - input_vals
        h2O = obst_pos - input_vals
        return np.concatenate( (h2t, h2O), axis = 2)
    else:
        h2t = goal_pos - input_vals
        d = np.sqrt( np.sum( np.multiply(obst_pos - input_vals, obst_pos - input_vals), axis=2))
        # d_star = rep_dist
        # print "rep_dist: " + str(type(rep_dist)) + str(rep_dist)
        sh = input_vals.shape
        d_star = np.repeat( np.array([[rep_dist]]), sh[1], axis=1)

        rep = ( np.reciprocal(d_star) - np.reciprocal(d) ) * np.reciprocal(d) * (obst_pos - input_vals)
        if np.squeeze( d < d_star):
            rep = rep
        else:
            rep = rep*0.0
        return np.concatenate( (h2t, rep), axis = 2)

def map_inputs(inpt, max_in, min_in):
    # Function to apply mapping: map non-normalized data to [-1,1]
    # inpt:             Unscaled network inputs
    # max_in:           Maximum values in original input data
    # min_in:           Minimum values in original input data
    sh = inpt.shape
    steps = sh[1]
    max_in = np.transpose( np.expand_dims(np.repeat(max_in, steps, axis=1), axis=0), (0,2,1) )
    min_in = np.transpose( np.expand_dims(np.repeat(min_in, steps, axis=1), axis=0), (0,2,1) )
    scaling = max_in - 0.5*(max_in + min_in)
    inpt_ = inpt - 0.5*(max_in+min_in)
    inpt_ = inpt_/scaling
    # print "Input: " + str(inpt_)
    return inpt_

def map_outputs(outp, max_out, min_out):
    # Function to reverse mapping for training outputs: map normalized data from [-1,1] to the original size
    # outp:             Scaled output values
    # max_out:          Maximum values in original output data
    # min_out:          Minimum values in original output data
    sh = outp.shape
    steps = sh[1]
    max_out = np.transpose( np.expand_dims(np.repeat(max_out, steps, axis=1), axis=0), (0,2,1) )
    min_out = np.transpose( np.expand_dims(np.repeat(min_out, steps, axis=1), axis=0), (0,2,1) )
    # print "Max_out: " + str(max_out.shape) + str(min_out.shape)
    scaling = max_out - 0.5*(max_out+min_out)
    outp_ = outp * scaling
    outp_ = outp_ + 0.5*(max_out + min_out)
    # print "Output: " + str(outp_)
    return outp_

def plotting_method(w_sub, o_sub, g_sub, loc, iB, tB, out, acc, vel, raw_sim_in, mx_out, mn_out, \
    idx, config_name, dt, add_noise, maxTsOrig, max_ts, last_val_idx, last_orig_seq, stat_sim):
        # Function in order to execute a number of predefinec plotting operations
        # w_sub:            Original wrist trajectory (positions)
        # o_sub:            Obstacle trajectory (position)
        # g_sub:            Goal trajectory (position)
        # loc:              Simulated wrist trajectory (positions)
        # tB:               targets (accelerations)
        # out:              predictions (acclerations) without feedback
        # acc:              predictions (accelerations) with feedback
        # mx_out:           Maximum values original data: outputs for scaling
        # min_out:          Minimum values original data: outputs for scaling

        # print "Acc-Shape: " + str(acc.shape)

        base_path = "Comparison/plots/" + config_name
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        base_path = base_path + "/" + str(idx)

        if add_noise:
            base_path = base_path + "_noise_"
            title_suffix = '[add. noise]'
        else:
            title_suffix = ''

        if stat_sim:
            base_path = base_path + "_static"
            title_suffix = ['STATIC']
        else:
            pass

        step_diff = loc.shape[0] - g_sub.shape[0]
        g_enlarg = np.concatenate((g_sub, np.matlib.repmat(g_sub[-1,:], step_diff, 1)), axis=0)
        o_enlarg = np.concatenate((o_sub, np.matlib.repmat(o_sub[-1,:], step_diff, 1)), axis=0)

        dist_h2g_orig = np.sqrt( \
            np.sum( np.multiply( np.squeeze(w_sub) - g_sub, np.squeeze(w_sub) - g_sub ), axis = 1 ))
        dist_h2o_orig = np.sqrt( \
            np.sum( np.multiply( np.squeeze(w_sub) - o_sub, np.squeeze(w_sub) - o_sub ), axis = 1 ))
        dist_h2o_sim = np.sqrt( \
            np.sum( np.multiply( loc - o_enlarg, loc - o_enlarg ), axis = 1 ))
        dist_h2g_sim = np.sqrt( \
            np.sum( np.multiply( loc - g_enlarg, loc - g_enlarg ), axis = 1 ))
        dist_h2o_sim_xy = np.sqrt( \
            np.sum( np.multiply( loc[:,0:2] - o_enlarg[:,0:2], loc[:,0:2] - o_enlarg[:,0:2] ), axis = 1 ))
        dist_h2g_sim_xy = np.sqrt( \
            np.sum( np.multiply( loc[:,0:2] - g_enlarg[:,0:2], loc[:,0:2] - g_enlarg[:,0:2] ), axis = 1 ))

        
        # Plot several distances for wrist to obstacle/goal in simulation and orig. recording
        """
        fig = plt.figure()
        splot = fig.add_subplot(111)
        sub_no_steps_1 = w_sub.shape[1]
        sub_no_steps_2 = loc.shape[0]
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), dist_h2g_orig, label='Wrist-to-goal dist: 3d, orig' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), dist_h2o_orig, label='Wrist-to-wrist dist: 3d, orig')
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), dist_h2g_sim, label='Wrist-to-goal dist: 3d, sim')
        hndl4, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), dist_h2o_sim, label='Wrist-to-wrist dist: 3d, sim')
        hndl5, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), dist_h2g_sim_xy, label='Wrist-to-goal dist: 2d, sim')
        hndl6, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), dist_h2o_sim_xy, label='Wrist-to-wrist dist: 2d, sim') 
        splot.set_xlabel(r'$[s]$', fontsize=16) # x-y switched
        splot.set_ylabel(r'$[mm]$', fontsize=16)
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6])
        fig.suptitle(r'$\mathbf{Distance\ metrics\ for\ original\ and}$' + '\n' + r'$\mathbf{ simulated\ scenario\ %s }$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.125, right=0.9, bottom = 0.125, top=0.85)
        plt.savefig(base_path + "_distances.eps")
        plt.savefig(base_path + "_distances.svg")
        plt.close()

        # Plot Original 2D wrist movement
        fig = plt.figure()
        splot = fig.add_subplot(111)
        hndl1, = splot.plot(w_sub[0,:,1], w_sub[0,:,0], label='Original wrist coord' )
        hndl2, = splot.plot(o_sub[:,1], o_sub[:,0], 'go', label='Obstacle 2nd wrist')
        hndl3, = splot.plot(o_sub[-1,1], o_sub[-1,0], 'bx', label='Fin. obstacle')
        hndl4, = splot.plot(g_sub[:,1], g_sub[:,0], 'ro', label='Goal')
        hndl5, = plt.plot(loc[:,1], loc[:,0], label='Simulated wrist coord' )
        splot.set_xlabel(r'$y \ [mm]$', fontsize=16) # x-y switched
        splot.set_ylabel(r'$x \ [mm]$', fontsize=16)
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5])
        fig.suptitle(r'$\mathbf{Wrist\ movements\ 2D\ %s}$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.125, right=0.9, bottom = 0.125, top=0.85)        # plt.show()
        plt.savefig(base_path + "_orig_sim_wrist.eps")
        plt.savefig(base_path + "_orig_sim_wrist.svg")
        plt.close()

        # Plot FFT accelerations, part
        fig = plt.figure()
        splot = fig.add_subplot(111)
        # np.fft.fft(acc[last_val_idx:,0])
        hndl1, = splot.plot(np.absolute(np.fft.rfft(acc[0, last_val_idx-1:,0])), label='FFT acc. x-dir' )
        hndl2, = splot.plot(np.absolute(np.fft.rfft(acc[0, last_val_idx-1:,1])), label='FFT acc. y-dir')
        hndl3, = splot.plot(np.absolute(np.fft.rfft(acc[0, last_val_idx-1:,2])), label='FFT acc. z-dir')
        splot.set_xlabel('FFT frequency (Ts = 0.32 ms)', fontsize=16) # x-y switched
        splot.set_ylabel('|FFT(f)|', fontsize=16)
        splot.legend(handles=[hndl1, hndl2, hndl3])
        fig.suptitle(r'$\mathbf{Accleration\ single \ coordinates}$' + '\n' + r'$\mathbf{FFT\ (part. sequence)\ %s }$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.13, right=0.9, bottom = 0.125, top=0.85)        # plt.show()
        plt.savefig(base_path + "_sim_acc_fft.eps")
        plt.savefig(base_path + "_sim_acc_fft.svg")
        plt.close()

        fig = plt.figure()
        splot = fig.add_subplot(111)
        # np.fft.fft(acc[last_val_idx:,0])
        hndl1, = splot.plot(np.absolute(np.fft.rfft(acc[0, :, 0])), label='FFT acc. x-dir' )
        hndl2, = splot.plot(np.absolute(np.fft.rfft(acc[0, :, 1])), label='FFT acc. y-dir')
        hndl3, = splot.plot(np.absolute(np.fft.rfft(acc[0, :, 2])), label='FFT acc. z-dir')
        splot.set_xlabel('FFT frequency (Ts = 0.32 ms)', fontsize=16) # x-y switched
        splot.set_ylabel(r'$|FFT(f)|$', fontsize=16)
        splot.legend(handles=[hndl1, hndl2, hndl3])
        fig.suptitle(r'$\mathbf{Accleration\ single \ coordinates}$' + '\n' + r'$\mathbf{FFT\ (full\ sequence)\ %s }$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.13, right=0.9, bottom = 0.125, top=0.85)        # plt.show()
        plt.savefig(base_path + "_sim_acc_full_fft.eps")
        plt.savefig(base_path + "_sim_acc_full_fft.svg")
        plt.close()

        
        # Plot FFT accelerations, full
        fig, splot = plt.subplots(4)
        # splot = fig.add_subplot(111)
        # np.fft.fft(acc[last_val_idx:,0])
        hndl1, = splot[0].plot(np.absolute(np.fft.rfft(acc[0, :, 0])), label='FFT acc. x-dir' )
        hndl2, = splot[1].plot(np.absolute(np.fft.rfft(acc[0, :, 1])), label='FFT acc. y-dir')
        hndl3, = splot[2].plot(np.absolute(np.fft.rfft(acc[0, :, 2])), label='FFT acc. z-dir')
        hndl4, = splot[3].plot(np.absolute(np.fft.rfft(np.sqrt(np.sum(np.squeeze(acc[0,:,:]*acc[0,:,:]),1)))), label='FFT acc total')
        for ax in splot:
            ax.set_xlabel(r'$FFT frequecy (Ts = 0.32 ms)$', fontsize=16) # x-y switched
            ax.set_ylabel(r'$|FFT(f)|$', fontsize=16)
        # splot.legend(handles=[hndl1, hndl2, hndl3])
        fig.suptitle(r'$\mathbf{Accleration\ single \ coordinates}$' + '\n' + r'$\mathbf{FFT\ (full sequence)\ %s }$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.125, right=0.9, bottom = 0.125, top=0.85)        # pjlt.show()
        plt.savefig(base_path + "_sim_acc_full_fft.eps")
        plt.close()
        
        

        # Plot simulated + original 2D wrist movement
        hndl1, = plt.plot(loc[:,1], loc[:,0], label='Simulated wrist coord' )
        hndl2, = plt.plot(o_sub[:,1], o_sub[:,0], 'go', label='Obstacle')
        hndl3, = plt.plot(o_sub[-1,1], o_sub[-1,0], 'x', label='Fin. obstacle')
        hndl4, = plt.plot(g_sub[:,1], g_sub[:,0], 'ro', label='Goal')
        plt.xlabel('y [mm]', fontsize=16) # x-y switched
        plt.ylabel('x [mm]', fontsize=16)
        plt.legend(handles=[hndl1, hndl2, hndl3, hndl4])
        # plt.show()
        plt.savefig(base_path + "_sim_wrist.eps")
        plt.close()
        
        # Plot UNSCALED forward prediction vs. original target acceleration
        hndl1, = plt.plot(tB[0,:,0], label='target acc x' )
        hndl2, = plt.plot(tB[0,:,1], label='target acc y' )
        #hndl3, = plt.plot(tB[0,:,2], label='target batch 3' )
        hndl4, = plt.plot(out[0,:,0], label='predict. acc. x')
        hndl5, = plt.plot(out[0,:,1], label='predict. acc. y')
        #hndl6, = plt.plot(out[0,:,2], label='prediction 3')
        #plt.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6])
        plt.ylabel('acc [mm/s^2]', fontsize=16)
        plt.xlabel('time [0.012s steps]', fontsize=16)
        plt.legend(handles=[hndl1, hndl2, hndl4, hndl5])
        # plt.show()
        plt.savefig(base_path + "_predicted_acc.eps")
        plt.close()
        

        # Plot velocities
        
        fig = plt.figure()
        splot = fig.add_subplot(111)
        sub_no_steps = vel.shape[1]
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), vel[0,:,0], label='Velocity x-dir' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps),vel[0,:,1], label='Velocity y-dir' )
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps),vel[0,:,2], label='Velocity z-dir' )
        splot.legend(handles=[hndl1, hndl2, hndl3])
        splot.set_ylabel(r'$[\frac{mm}{s}]$', fontsize=16)
        splot.set_xlabel(r'[s]', fontsize=16)
        fig.suptitle(r'$\mathbf{Directional\ end}$' + '\n' + r'$\mathbf{effector\ velocities\ %s }$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.16, right=0.9, bottom = 0.125, top=0.85)
        # plt.show()
        plt.savefig(base_path + "_sim_vel.eps")
        plt.close()

        
 
        # Plot SCALED feedback outputs
        tB = map_outputs(tB, mx_out, mn_out)
        sub_no_steps_1 = tB.shape[1]
        sub_no_steps_2 = acc.shape[1]
        fig = plt.figure()
        splot = fig.add_subplot(111)
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,0], label='Target acc. x' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,1], label='Target acc. y' )
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,2], label='Target acc. z' )
        hndl4, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,0], label='Predict. acc. fb. x')
        hndl5, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,1], label='Predict. acc. fb. y')
        hndl6, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,2], label='Predict. acc. fb. z')
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6])
        splot.set_ylabel(r'$[\frac{mm}{s^2}]$', fontsize=16)
        splot.set_xlabel(r'$[s]$', fontsize=16)
        fig.suptitle(r'$\mathbf{Directional\ end\ effector}$' + '\n' r'$\mathbf{ accelerations\ %s}$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.16, right=0.9, bottom = 0.125, top=0.85)
        plt.savefig(base_path + "_sim_acc.eps")
        plt.savefig(base_path + "_sim_acc.svg")
        plt.close()

        
        # Plot SCALED feedback outputs
        fig = plt.figure()
        splot = fig.add_subplot(111)
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,0], color='C0', label='Target acc. x' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,1], linestyle='--', color='black', label='Target acc. y' )
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,2], linestyle=':', color='black', label='Target acc. z' )
        hndl4, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,0], color='C3', label='Predict. acc. fb. x')
        hndl5, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,1], linestyle='--', color='gray', label='Predict. acc. fb. y')
        hndl6, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,2], linestyle=':', color='gray', label='Predict. acc. fb. z')
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6])
        splot.set_ylabel(r'$[\frac{mm}{s^2}]$', fontsize=16)
        splot.set_xlabel(r'$[s]$', fontsize=16)
        fig.suptitle(r'$\mathbf{Directional\ end\ effector}$' + '\n' r'$\mathbf{ accelerations\ %s}$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.16, right=0.9, bottom = 0.125, top=0.85)
        plt.savefig(base_path + "_sim_acc_x.eps")
        plt.savefig(base_path + "_sim_acc_x.svg")
        plt.close()

        # Plot SCALED feedback outputs
        fig = plt.figure()
        splot = fig.add_subplot(111)
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,0], linestyle='--', color='black', label='Target acc. x' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,1], color='C1',label='Target acc. y' )
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,2], linestyle=':', color='black', label='Target acc. z' )
        hndl4, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,0], linestyle='--', color='gray', label='Predict. acc. fb. x')
        hndl5, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,1], color='C4',label='Predict. acc. fb. y')
        hndl6, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,2], linestyle=':', color='gray', label='Predict. acc. fb. z')
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6])
        splot.set_ylabel(r'$[\frac{mm}{s^2}]$', fontsize=16)
        splot.set_xlabel(r'$[s]$', fontsize=16)
        fig.suptitle(r'$\mathbf{Directional\ end\ effector}$' + '\n' r'$\mathbf{ accelerations\ %s}$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.16, right=0.9, bottom = 0.125, top=0.85)
        plt.savefig(base_path + "_sim_acc_y.eps")
        plt.savefig(base_path + "_sim_acc_y.svg")
        plt.close()

        # Plot SCALED feedback outputs
        fig = plt.figure()
        splot = fig.add_subplot(111)
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,0], linestyle='--', color='black', label='Target acc. x' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,1], linestyle=':', color='black',label='Target acc. y' )
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), tB[0,:,2], color='C2', label='Target acc. z' )
        hndl4, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,0], linestyle='--', color='gray',label='Predict. acc. fb. x')
        hndl5, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,1], linestyle=':', color='gray',label='Predict. acc. fb. y')
        hndl6, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), acc[0,:,2], color='C5', label='Predict. acc. fb. z')
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6])
        splot.set_ylabel(r'$[\frac{mm}{s^2}]$', fontsize=16)
        splot.set_xlabel(r'$[s]$', fontsize=16)
        fig.suptitle(r'$\mathbf{Directional\ end\ effector}$' + '\n' r'$\mathbf{ accelerations\ %s}$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.16, right=0.9, bottom = 0.125, top=0.85)
        plt.savefig(base_path + "_sim_acc_z.eps")
        plt.savefig(base_path + "_sim_acc_z.svg")
        plt.close()
        
        

        
        # Plot original raw inputs:
        fig = plt.figure()
        splot = fig.add_subplot(111)
        sub_no_steps = iB.shape[1]
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), iB[0,:,0], label='Orig. in H2T x' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), iB[0,:,1], label='Orig. in H2T y' )
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), iB[0,:,2], label='Orig. in H2T z' )
        hndl4, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), iB[0,:,3], label='Orig. in Rep x')
        hndl5, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), iB[0,:,4], label='Orig. in Rep y')
        hndl6, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), iB[0,:,5], label='Orig. in Rep z')
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6])
        splot.set_ylabel(r'$Relative\ and\ scaled$' + '\n' + r'$distance\ units$', fontsize=16)
        splot.set_xlabel(r'$[s]$', fontsize=16)
        fig.suptitle(r'$\mathbf{Original\ raw\ network\ inputs\ %s}$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.16, right=0.9, bottom = 0.125, top=0.85)
        plt.savefig(base_path + "_inputs_raw_orig.eps")
        plt.close()

        # Plot simulated raw inputs:
        fig = plt.figure()
        splot = fig.add_subplot(111)
        sub_no_steps = raw_sim_in.shape[1]
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), raw_sim_in[0,:,0], label='Sim. in H2T x' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), raw_sim_in[0,:,1], label='Sim. in H2T y' )
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), raw_sim_in[0,:,2], label='Sim. in H2T z' )
        hndl4, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), raw_sim_in[0,:,3], label='Sim. Rep. x')
        hndl5, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), raw_sim_in[0,:,4], label='Sim. Rep. y')
        hndl6, = splot.plot(np.linspace(0,dt*(sub_no_steps-1),sub_no_steps), raw_sim_in[0,:,5], label='Sim. Rep. z')
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6])
        splot.set_ylabel(r'$Relative\ and\ scaled$' + '\n' + r'$distance\ units$', fontsize=16)
        splot.set_xlabel(r'$[s]$', fontsize=16)
        fig.suptitle(r'$\mathbf{Simulation\ raw\ network}$' + '\n' + r'$\mathbf{inputs\ %s}$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.16, right=0.9, bottom = 0.125, top=0.85)
        plt.savefig(base_path + "_inputs_raw_sim.eps")
        plt.close()

        # Plot single wrist coordinates: original vs. fb-simulated
        fig = plt.figure()
        splot = fig.add_subplot(111)
        sub_no_steps_1 = w_sub.shape[1]
        sub_no_steps_2 = loc.shape[0]
        hndl1, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), w_sub[0,:,0], label='Wrist orig. x' )
        hndl2, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), w_sub[0,:,1], label='Wrist orig. y' )
        hndl3, = splot.plot(np.linspace(0,dt*(sub_no_steps_1-1),sub_no_steps_1), w_sub[0,:,2], label='Wrist orig. z' )
        hndl4, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), loc[:,0], label='Wrist sim. fb x')
        hndl5, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), loc[:,1], label='Wrist sim. fb y')
        hndl6, = splot.plot(np.linspace(0,dt*(sub_no_steps_2-1),sub_no_steps_2), loc[:,2], label='Wrist sim. fb z')
        splot.legend(handles=[hndl1, hndl2, hndl3, hndl4, hndl5, hndl6], loc=7)
        splot.set_ylabel(r'$[mm]$', fontsize=16)
        splot.set_xlabel(r'$[s]$', fontsize=16)
        fig.suptitle(r'$\mathbf{Single\ directional\ positions\ of\ the }$' + '\n' + r'$\mathbf{end\ effector/wrist \ %s }$'%(title_suffix), fontsize=18)
        plt.subplots_adjust(left=0.125, right=0.9, bottom = 0.125, top=0.85)
        plt.savefig(base_path + "_single_wrist.eps")
        plt.savefig(base_path + "_single_wrist.svg")
        plt.close()
        

        # Plot single wrist coordinates: original vs. fb-simulated
        # c.f. https://stackoverflow.com/questions/11541123/how-can-i-make-a-simple-3d-line-with-matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        hndl1, = ax.plot(w_sub[0,:,0], w_sub[0,:,1], w_sub[0,:,2], label='P1 hand')
        hndl2, = ax.plot(loc[:,0], loc[:,1], loc[:,2], label='End effector simulation')
        hndl3, = ax.plot(o_sub[:,0], o_sub[:,1], o_sub[:,2], label='P2 hand')
        # hndl4, = ax.plot(o_sub[-1,0], o_sub[-1,1], o_sub[-1,2], 'x', label='Fin. obstacle')
        ax.legend(handles=[hndl1, hndl2, hndl3])
        ax.tick_params(axis='x', which='major', pad=-5)
        ax.set_xlabel(r'$x\ [mm]$', fontsize=16)
        ax.xaxis.labelpad = -3
        ax.tick_params(axis='y', which='major', pad=-5)
        ax.set_ylabel(r'$y\ [mm]$', fontsize=16)
        ax.yaxis.labelpad = -3
        ax.tick_params(axis='z', which='major', pad=-3)
        ax.set_zlabel(r'$z\ [mm]$', fontsize=16)
        ax.zaxis.labelpad = -3
        fig.suptitle(r'$\mathbf{Hand\ movements\ 3D\ %s}$'%(title_suffix), fontsize=18)
        plt.savefig(base_path + "_wrist_3d.eps")
        # plt.savefig(base_path + "_wrist_3d.svg")
        plt.close()
        """
        
        

        # Now save all the relevant values to the csv_file
        Max_vel_x = np.amax( np.squeeze(vel[0,:,0]) )
        Max_vel_y = np.amax( np.squeeze(vel[0,:,1]) )
        Max_vel_z = np.amax( np.squeeze(vel[0,:,2]) )
        Min_vel_x = np.amin( np.squeeze(vel[0,:,0]) )
        Min_vel_y = np.amin( np.squeeze(vel[0,:,1]) )
        Min_vel_z = np.amin( np.squeeze(vel[0,:,2]) )
        vel_tot = np.sqrt(np.sum(np.squeeze(vel[0,:,:]*vel[0:,:]),1))
        Max_vel_total = np.amax(vel_tot)
        Min_vel_total = np.amin(vel_tot)

        Max_acc_x = np.amax( np.squeeze(acc[0,:,0]) )
        Max_acc_y = np.amax( np.squeeze(acc[0,:,1]) )
        Max_acc_z = np.amax( np.squeeze(acc[0,:,2]) )
        Min_acc_x = np.amin( np.squeeze(acc[0,:,0]) )
        Min_acc_y = np.amin( np.squeeze(acc[0,:,1]) )
        Min_acc_z = np.amin( np.squeeze(acc[0,:,2]) )
        acc_tot = np.sqrt(np.sum(np.squeeze(acc[0,:,:]*acc[0:,:]),1))
        Max_acc_total = np.amax(acc_tot)
        Min_acc_total = np.amin(acc_tot)

        Max_pos_x = np.amax( np.squeeze(loc[:,0]) )
        Max_pos_y = np.amax( np.squeeze(loc[:,1]) )
        Max_pos_z = np.amax( np.squeeze(loc[:,2]) )
        Min_pos_x = np.amin( np.squeeze(loc[:,0]) )
        Min_pos_y = np.amin( np.squeeze(loc[:,1]) )
        Min_pos_z = np.amin( np.squeeze(loc[:,2]) )
        pos_tot = np.sqrt(np.sum(np.squeeze(loc[:,:]*loc[:,:]),1))
        Max_pos_total = np.amax(pos_tot)
        Min_pos_total = np.amin(pos_tot)

        Min_H2O_dist = np.amin(dist_h2o_sim)

        threshold_reach_step = last_val_idx

        sim_time = loc.shape[0]

        MSS_FFT_full_x = np.mean( np.square( np.absolute(np.fft.rfft(acc[0, :, 0])) ) ) # energy according to Parseval theorem
        MSS_FFT_full_y = np.mean( np.square( np.absolute(np.fft.rfft(acc[0, :, 1])) ) ) # energy according to Parseval theorem
        MSS_FFT_full_z = np.mean( np.square( np.absolute(np.fft.rfft(acc[0, :, 2])) ) ) # energy according to Parseval theorem
        MSS_FFT_full_total = np.mean( np.square( np.absolute( np.fft.rfft(np.sqrt(np.sum(np.squeeze(acc[0,:,:]*acc[0,:,:]),1))))) )

        orig_time = last_orig_seq

        terminal_h2g_dist = dist_h2g_sim[-1]
        terminal_h2g_dist_xy = dist_h2g_sim_xy[-1]

        print vel_tot[10:].shape
        termination_step_vel = np.argmax(vel_tot[10:] < 50) # alternative stopping criterion 50 mm/s

        if not last_val_idx == sim_time:
            MSS_FFT_part_x = np.mean( np.square( np.absolute(np.fft.rfft(acc[0, last_val_idx-1:,0]) ) ) )
            MSS_FFT_part_y = np.mean( np.square( np.absolute(np.fft.rfft(acc[0, last_val_idx-1:,1]) ) ) )
            MSS_FFT_part_z = np.mean( np.square( np.absolute(np.fft.rfft(acc[0, last_val_idx-1:,2]) ) ) )
            MSS_FFT_part_total = np.mean( np.square( np.absolute(np.fft.rfft(np.sqrt(np.sum(np.squeeze(acc[0,last_val_idx-1:,:]*acc[0,last_val_idx-1:,:]),1))) ) ) )
        else:
            MSS_FFT_part_x = None
            MSS_FFT_part_y = None
            MSS_FFT_part_z = None
            MSS_FFT_part_total = None

        csv_file_prefix = "Comparison/numerical_data/" + config_name
        if add_noise:
            csv_file_prefix = csv_file_prefix + "_noise"
        else:
            pass

        if stat_sim:
            csv_file_prefix = csv_file_prefix = "_static"
        else:
            pass

        with open(csv_file_prefix + ".csv", 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([Max_vel_x, Max_vel_y, Max_vel_z, Max_vel_total, Min_vel_x, Min_vel_y, Min_vel_z, Min_vel_total, \
                Max_acc_x, Max_acc_y, Max_acc_z, Max_acc_total, Min_acc_x, Min_acc_y, Min_acc_z, Min_acc_total, \
                Max_pos_x, Max_pos_y, Max_pos_z, Max_pos_total, Min_pos_x, Min_pos_y, Min_pos_z, Min_pos_total, \
                Min_H2O_dist, threshold_reach_step, sim_time, MSS_FFT_full_x, MSS_FFT_full_y, MSS_FFT_full_z, MSS_FFT_full_total, \
                MSS_FFT_part_x, MSS_FFT_part_y, MSS_FFT_part_z, MSS_FFT_part_total, orig_time, terminal_h2g_dist, terminal_h2g_dist_xy, termination_step_vel])

def load_test_setup(load_file_name):
    # Deprecated
    # Loads a test setup in a predefine format (see below)
    # load_file_name:           the path/file name to load the data from
    npzfile = np.load(load_file_name)
    loc = npzfile['loc']
    w_sub = npzfile['w_sub']
    acc = npzfile['acc']
    out = npzfile['out']
    tB = npzfile['tB']
    vel = npzfile['vel']
    g_sub = npzfile['g_sub']
    o_sub = npzfile['o_sub']
    mx_in = npzfile['mx_in']
    mn_in = npzfile['mn_in']
    mx_out = npzfile['mx_out']
    mn_out = npzfile['mn_out']
    print "Results loaded"
    return w_sub, o_sub, g_sub, loc, tB, out, acc, mx_in, mn_in, mx_out, mn_out

def truncate_and_interpolate_sequence(pos, targ, num_steps, threshold, maxTsOrig):
    # This function computest the truncated trajectory that satisfies a maximum distance to the goal up to the threshold value
    # In a second step this sequence is interpolated and fitted to a sequence length of 51
    # 
    # pos:              Output position (from NN) [550,3]
    # targ:             Target position
    # num_steps:        Required steps in the ProMP model
    # threshold:        Threshold to determine goal reaching for NN-outputs (pos)
    # maxTsOrig:        Maximum time steps in original data
    # maxTs:            Maximum time steps in simulation

    # adjust shape of target
    targ = np.vstack( (targ, np.matlib.repmat(targ[-1,:], num_steps-maxTsOrig, 1)) )
    dist = np.sqrt( np.sum( (pos - targ)*(pos - targ), axis=1) )
    satisf_elements = np.less(dist, threshold)
    rev_not_satisfied_sequence = np.bitwise_not( np.equal(np.cumsum(np.flipud(satisf_elements)), np.arange(1,num_steps+1)) )
    last_valid_index = np.sum(rev_not_satisfied_sequence)
    valid_sequence = pos[0:last_valid_index]
    print "Shape of valid squence: " + str(valid_sequence.shape) + "with last valid index" + str(last_valid_index)
    # print np.sum(rev_not_satisfied_sequence)

    if valid_sequence.shape[0] > 0:
        # print "Export truncated and stretched sequence"
        interpolated_sequence_x = np.interp(np.linspace(1,last_valid_index,51), np.linspace(1,last_valid_index, last_valid_index), valid_sequence[:,0])
        interpolated_sequence_y = np.interp(np.linspace(1,last_valid_index,51), np.linspace(1,last_valid_index, last_valid_index), valid_sequence[:,1])
        interpolated_sequence_z = np.interp(np.linspace(1,last_valid_index,51), np.linspace(1,last_valid_index, last_valid_index), valid_sequence[:,2])
        extracted_trajectory = np.vstack((interpolated_sequence_x, interpolated_sequence_y, interpolated_sequence_z))
        return extracted_trajectory, last_valid_index
    else:
        raise ValueError('Threshold condition must be valid for a minimum of one steps')

def export_data_kl_divergence(cluster_trajectories, forward_idxs, backward_idxs, other_idxs,type_flag, config_name, add_noise):
    # This function takes a cluster of trajectories, sorts it according to forward and backward movement and saves .mat files 
    # in a specific format to feed into the ProMP-Generator-Mechanism [# Trajectories x # Time steps x # Dimensions]
    # 
    # cluster_trajectories: Cluster of all trajectories
    # forward_idxs:         Indices that indicate which trajectories are forward movements (may be defined manually)
    # backward_idxs:        Indices that indicate which trajectories are backward movements (may be defined manually)
    # other_idxs:           Indices that indicate which trajectories are other movements (may be defined manuallydx)
    base_path = 'Comparison/ProMPData/' + config_name + '/' + type_flag
    if add_noise:
        base_path = base_path + '/with_noise'
    else:
        base_path = base_path + '/without_noise'
    if not os.path.exists(base_path):
        os.makedirs(base_path + '/forward/')
        os.makedirs(base_path + '/backward/')
    sio.savemat(base_path + '/forward/training_dataset.mat', {'TrajList': cluster_trajectories[forward_idxs,:,:]})
    sio.savemat(base_path + '/backward/training_dataset.mat', {'TrajList': cluster_trajectories[backward_idxs,:,:]})
    return True

def visualize_variable_matrices(variables_list, config_name):
    # This function visualizes the model variables to see neuron impacts
    # variables_list:           list of all trainable variables
    base_path = "Comparison/plots/" + config_name + "/"
    for idx, var_ in enumerate(variables_list):
        var = var_.eval()
        if len(var.shape) == 2:
            name = copy.copy(var_.name)
            name = name.replace("/","_")
            name = name.replace(":", "_")
            # http://seaborn.pydata.org/generated/seaborn.heatmap.html
            fig = plt.figure()
            splot = fig.add_subplot(111)
            ax = sns.heatmap(var, annot=True)
            # https://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
            fig.set_size_inches(16, 9)
            fig.suptitle(name)
            plt.savefig(base_path + "0" + name + "_variables" + ".eps", format="eps")
            # plt.show()
            plt.close()
        elif len(var.shape) == 1:
            pass

    
class neuron_states(object):
    # This is a class that defines an object for user controlled batch-fetching
    def __init__(self, netw_state, layer_list, neuron_list, add_noise=False):
        # Call as constructor
        # layer_list:           List of each layer's type
        # neuron_list           List of neurons per layer
        # netw_state:           Network state used to initialize the state list
        self.layer_list = layer_list
        self.neuron_list = neuron_list
        self.state_dict = dict()
        self.add_noise = add_noise
        self.no_memory_states = 0 # number of memory states in network (2 per layer in LSTM-like, 1 otherwise)
        self.no_single_states = 0
        for idx, tp in enumerate(layer_list):
            self.state_dict[str(idx)] =self.generate_state_entry(tp, netw_state[idx], idx)

    def get_state_dict(self,idx=0):
        return self.state_dict[str(idx)]

    def get_no_single_states(self):
    	return self.no_single_states

    def generate_state_entry(self, layer_type, lay_state, layer_idx):
        # This function generates an entry in the state dict and also counts total memory states
        #
        # layer_type:           Type of the respective layer in the network
        # lay_state:            State of the respective layer in the network
        if layer_type in ["Basic", "LSTM", "BasicMod", "BasicMod2"]:
            self.no_memory_states = self.no_memory_states + 2
            self.no_single_states = self.no_single_states + self.neuron_list[layer_idx]*2
            return {'c': lay_state.c, 'h': lay_state.h}
        else:
            self.no_memory_states = self.no_memory_states + 1
            self.no_single_states = self.no_single_states + self.neuron_list[layer_idx]
            return lay_state

    def stack_data(self, netw_state):
        # This function adds all isolated network states for each time step int the state_dict
        # 
        # netw_state:       Network state for the respective time step
        for idx, tp in enumerate(self.layer_list):
            if tp in ["Basic", "LSTM", "BasicMod", "BasicMod2"]:
                self.state_dict[str(idx)]['c'] = np.vstack((self.state_dict[str(idx)]['c'], netw_state[idx].c))
                self.state_dict[str(idx)]['h'] = np.vstack((self.state_dict[str(idx)]['h'], netw_state[idx].h))
            else:
                self.state_dict[str(idx)] = np.vstack((self.state_dict[str(idx)], netw_state[idx]))
        return True

    def correlate_states_with_quantities(self, quantity, name, max_ts):
        # This function computes the correlation of single states with different coordinates of
        # 
        # quantity:         Quantity to correlate against must be of same size as states. Requires 2D shape
        #                   Use np.squeeze if dim = 3,4,5....

        # Create placeholder of each correlation value: i.e. quantity dimensions e.g. 3 dim x num neurons x memory states
        corr_matrix = np.zeros((self.no_memory_states*max(self.neuron_list), quantity.shape[1]), dtype=np.float64)
        mem_state_idx = 0
        for idx, tp in enumerate(self.layer_list):
            for state_idx in range(self.neuron_list[idx]):
                for dim_idx in range(quantity.shape[1]):
                    # print idx, state_idx, dim_idx, mem_state_idx
                    if tp in ["Basic", "LSTM", "BasicMod", "BasicMod2"]:
                        corr_matrix[mem_state_idx*max(self.neuron_list)+state_idx, dim_idx] = np.correlate( \
                            self.state_dict[str(idx)]['c'][0:max_ts,state_idx], quantity[0:max_ts,dim_idx])/ \
                        (np.linalg.norm(self.state_dict[str(idx)]['c'][0:max_ts,state_idx])  * np.linalg.norm(quantity[0:max_ts,dim_idx])) 
                        corr_matrix[(mem_state_idx+1)*max(self.neuron_list)+state_idx, dim_idx] = np.correlate( \
                            self.state_dict[str(idx)]['h'][0:max_ts,state_idx], quantity[0:max_ts,dim_idx])/ \
                        (np.linalg.norm(self.state_dict[str(idx)]['h'][0:max_ts,state_idx])  * np.linalg.norm(quantity[0:max_ts,dim_idx])) 
                        if state_idx == self.neuron_list[idx]-1 and dim_idx == quantity.shape[1]-1:
                            mem_state_idx = mem_state_idx + 2
                    else:
                        corr_matrix[mem_state_idx*max(self.neuron_list)+state_idx, dim_idx] = np.correlate( \
                            self.state_dict[str(idx)][0:max_ts,state_idx], quantity[0:max_ts,dim_idx]) / \
                        (np.linalg.norm(self.state_dict[str(idx)][0:max_ts,state_idx])  * np.linalg.norm(quantity[0:max_ts,dim_idx])) 
                        if state_idx == self.neuron_list[idx]-1 and dim_idx == quantity.shape[1]-1:
                            mem_state_idx = mem_state_idx + 1
        return corr_matrix

    def plot_data(self, config_name, dt, num_steps, stat_sim, global_idx=0):
        # This function handles the plots of the network states
        # 
        # num_steps:        Number of relevant time steps
        # global_idx:       Global test index for correct file naes

        base_path = "Comparison/plots/" + config_name + "/"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        base_path = base_path + str(global_idx)

        if self.add_noise:
            base_path = base_path + "_noise"
            title_suffix = '[add. noise]'
        else:
            title_suffix = ''

        if stat_sim:
            base_path = base_path + "_static"
            title_suffix = 'STATIC'
        else:
            pass

        for idx, tp in enumerate(self.layer_list):
            if tp in ["Basic", "LSTM", "BasicMod", "BasicMod2"]:
                fig = plt.figure()
                splot = fig.add_subplot(111)
                plots = [splot.plot(np.linspace(0,(num_steps-1)*dt, num_steps), self.state_dict[str(idx)]['c'][:,pidx], label=str(idx+1)) for pidx in range(self.neuron_list[idx])]
                splot.set_ylabel(r'$State\ amplitude$', fontsize=16)
                splot.set_xlabel(r'$[s]$', fontsize=16)
                fig.suptitle(r'$\mathbf{Amplitudes\ of\ single\ unit\ states}$' + '\n' + r'$\mathbf{for\ LSTM\ (%s \ type) \ c\ layer\ %s \ %s }$'%(tp, idx+1, title_suffix), fontsize=18)
                plt.subplots_adjust(left=0.125, right=0.9, bottom = 0.125, top=0.85)
                plt.savefig(base_path + "_layer_" + str(idx+1) + "_c.eps", format="eps")
                plt.close()

                fig = plt.figure()
                splot = fig.add_subplot(111)
                plots = [splot.plot(np.linspace(0,(num_steps-1)*dt, num_steps), self.state_dict[str(idx)]['h'][:,pidx], label=str(pidx+1)) for pidx in range(self.neuron_list[idx])]
                splot.set_ylabel(r'$State\ amplitude$', fontsize=16)
                splot.set_xlabel(r'$[s]$', fontsize=16)
                fig.suptitle(r'$\mathbf{Amplitudes\ of\ single\ unit\ states}$' + '\n' + r'$\mathbf{for\ LSTM\ (%s \ type) \ h\ layer\ %s \ %s}$'%(tp, idx+1, title_suffix), fontsize=18)
                plt.subplots_adjust(left=0.125, right=0.9, bottom = 0.125, top=0.85)
                plt.savefig(base_path + "_layer_" + str(idx+1) + "_h.eps", format="eps")
                plt.close()
            else:
                fig = plt.figure()
                splot = fig.add_subplot(111)
                # check if that works
                cm = plt.get_cmap('gist_rainbow')
                cNorm  = colors.Normalize(vmin=0, vmax=self.neuron_list[idx]-1)
                scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
                plots = [splot.plot(np.linspace(0,(num_steps-1)*dt, num_steps), self.state_dict[str(idx)][:,pidx], label=str(pidx+1), color=scalarMap.to_rgba(pidx)) for pidx in range(self.neuron_list[idx])]
                splot.set_ylabel(r'$State\ amplitude$', fontsize=16)
                splot.set_xlabel(r'$[s]$', fontsize=16)
                handles, labels = splot.get_legend_handles_labels()
                splot.legend(handles, labels)
                fig.suptitle(r'$\mathbf{Amplitudes\ of\ single\ unit\ states}$' + '\n' + r'$\mathbf{for\ %s \ layer\ %s \ %s}$'%(tp, idx+1, title_suffix), fontsize=18)
                plt.subplots_adjust(left=0.125, right=0.9, bottom = 0.125, top=0.85)
                plt.savefig(base_path + "_layer_" + str(idx+1) + ".eps", format="eps")
                plt.close()


