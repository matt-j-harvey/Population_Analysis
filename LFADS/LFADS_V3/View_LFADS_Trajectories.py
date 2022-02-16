import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l1, l2, L1L2
from keras import layers, Sequential, Input
from keras.layers import LSTM, Dense, GRU, ZeroPadding2D
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
import os

import LFADS_Model_V3
import Visualise_Model
import Import_Preprocessed_Data


def smooth_delta_f_matrix(delta_f_maxtrix):

    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_delta_f = []

    for trace in delta_f_maxtrix:
        smoothed_trace = np.convolve(trace, kernel, mode='same')
        smoothed_delta_f.append(smoothed_trace)

    smoothed_delta_f = np.array(smoothed_delta_f)
    return smoothed_delta_f



def normalise_delta_f_matrix(delta_f_matrix):

    delta_f_matrix = np.transpose(delta_f_matrix)
    # Normalise Each Neuron to Min 0, Max 1

    # Subtract Min To Get Min = 0
    min_vector = np.min(delta_f_matrix, axis=0)
    delta_f_matrix = np.subtract(delta_f_matrix, min_vector)

    # Divide By Max To Get Max = 1
    max_vector = np.max(delta_f_matrix, axis=0)
    delta_f_matrix = np.divide(delta_f_matrix, max_vector)

    delta_f_matrix = np.transpose(delta_f_matrix)
    delta_f_matrix = np.nan_to_num(delta_f_matrix)
    return delta_f_matrix


def create_trial_tensor(delta_f_matrix, onsets, start_window, stop_window):
    # Given A List Of Trial Onsets - Create A 3 Dimensional Tensor (Trial x Neuron x Trial_Aligned_Timepoint)

    number_of_timepoints = np.shape(delta_f_matrix)[1]

    # Transpose Delta F Matrix So Its Time x Neurons
    delta_f_matrix = np.transpose(delta_f_matrix)

    selected_data = []
    for onset in onsets:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_stop < number_of_timepoints:
            trial_data = delta_f_matrix[int(trial_start):int(trial_stop)]
            selected_data.append(trial_data)

    selected_data = np.array(selected_data)

    return selected_data



def load_session_data(matlab_file, condition_names, trial_start, trial_stop):

        # Create Matlab Object
        data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file)

        # Extract Delta F Matrix
        delta_f_matrix = data_object.dF
        delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
        delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)

        # Extract Frame Onsets
        condition_1_onsets = data_object.vis1_frames[0]
        condition_2_onsets = data_object.irrel_vis1_frames[0]
        expected_odour_trials = data_object.mismatch_trials['exp_odour'][0]

        # Create Trial Tensor
        condition_1_tensor = create_trial_tensor(delta_f_matrix, condition_1_onsets, trial_start, trial_stop)
        condition_2_tensor = create_trial_tensor(delta_f_matrix, condition_2_onsets, trial_start, trial_stop)
        full_tensor = np.concatenate([condition_1_tensor, condition_2_tensor])

        return condition_1_tensor, condition_2_tensor, full_tensor


def load_data(session_list, condition_names, trial_start, trial_stop):

    # Get Data Structure
    input_data = []
    session_neuron_numbers = []
    session_trial_numbers = []
    trial_length = 0

    for session in session_list:

        # Load Session Data
        condition_1_tensor, condition_2_tensor, full_tensor = load_session_data(session, condition_names, trial_start, trial_stop)

        # Get Session Structure
        session_trials = np.shape(full_tensor)[0]
        trial_length = np.shape(full_tensor)[1]
        session_neurons = np.shape(full_tensor)[2]


        input_data.append(full_tensor)
        session_neuron_numbers.append(session_neurons)
        session_trial_numbers.append(session_trials)

        print("Session", session, "Number Of Trials", session_trials, "Number of neurons", session_neurons)

    input_tensor = [input_data]
    input_tensor = tf.ragged.constant(input_tensor, dtype=tf.float32)

    return input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length







session_list = ["/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201022_112044__ACV004_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201016_113151__ACV004_B2_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201026_103629__ACV014_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201024_104327__ACV005_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201026_122511__ACV011_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201103_160924__ACV013_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201029_145825__ACV011_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201021_121703__ACV005_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201027_140620__ACV013_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20200922_114059__ACV003_B3_SWITCH_preprocessed_basic.mat"]

session_list = ["/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201022_112044__ACV004_B3_SWITCH_preprocessed_basic.mat",
                "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201016_113151__ACV004_B2_SWITCH_preprocessed_basic.mat"]

# Load Data
number_of_sessions = len(session_list)
condition_names = None
trial_start = -5
trial_stop = 15

input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length = load_data(session_list, condition_names, trial_start, trial_stop)



number_of_sessions = len(session_list)
input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length = load_data(session_list, condition_names, trial_start, trial_stop)

print("Data Shape", input_tensor.shape)
print("Session neuron numbers", session_neuron_numbers)
print("Session trial numbers", session_trial_numbers)

# Create Model
number_of_factors = 30
latent_dimensions = 3
model = LFADS_Model_V3.LFADS_Model(session_neuron_numbers, session_trial_numbers, number_of_factors, trial_length, number_of_sessions, latent_dimensions)


# Load Model Weights
weights_file = "/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Model_Weights/570_Model_weights"
model.load_weights(weights_file)


# Create Figure
figure_1 = plt.figure()
rows = 1
columns = 1
inferred_axis = figure_1.add_subplot(rows, columns, 1, projection='3d')

# View Infered Trajectoreis
Visualise_Model.view_trajectories(model, number_of_sessions, input_data, inferred_axis, 1, 1)
inferred_axis.set_title("Inferred Trajectories")

plt.show()