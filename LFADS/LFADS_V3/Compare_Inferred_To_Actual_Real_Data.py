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
import jPCA

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
    condition_1_tensor_list = []
    condition_2_tensor_list = []

    for session in session_list:

        # Load Session Data
        condition_1_tensor, condition_2_tensor, full_tensor = load_session_data(session, condition_names, trial_start, trial_stop)

        # Get Session Structure
        session_trials = np.shape(full_tensor)[0]
        trial_length = np.shape(full_tensor)[1]
        session_neurons = np.shape(full_tensor)[2]

        input_data.append(full_tensor)
        condition_1_tensor_list.append(condition_1_tensor)
        condition_2_tensor_list.append(condition_2_tensor)
        session_neuron_numbers.append(session_neurons)
        session_trial_numbers.append(session_trials)

        print("Session", session, "Number Of Trials", session_trials, "Number of neurons", session_neurons)

    input_tensor = [input_data]
    input_tensor = tf.ragged.constant(input_tensor, dtype=tf.float32)

    return input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length, condition_1_tensor_list, condition_2_tensor_list



def view_trajectories_jpca_interactive(model, number_of_sessions, condition_1_data_list, condition_2_data_list, trajectory_axis):


    # Get Encoder and Generator
    encoder = model.encoder
    generator = model.generator

    colourmap = cm.get_cmap('hsv')

    low_d_trajectory_list = []
    colour_list = []

    for session_index in range(number_of_sessions):

        # Input Translation Weight
        condition_1_data = condition_1_data_list[session_index]
        condition_2_data = condition_2_data_list[session_index]

        # Translate To Factors
        session_input_translation_weights = model.input_translation_weights_list[session_index]
        session_input_translation_weights = np.array(session_input_translation_weights)
        condition_1_data_factors = np.matmul(condition_1_data, session_input_translation_weights)
        condition_2_data_factors = np.matmul(condition_2_data, session_input_translation_weights)

        # Encoder To Initial States
        condition_1_z_mean, condition_1_z_log_var, condition_1_initial_states = encoder(condition_1_data_factors)
        condition_2_z_mean, condition_2_z_log_var, condition_2_initial_states = encoder(condition_2_data_factors)

        # Generate Low D Trajectories From Initial States
        condition_1_low_d_trajectories = generator(condition_1_initial_states)
        condition_2_low_d_trajectories = generator(condition_2_initial_states)

        for trajectory in condition_1_low_d_trajectories:
            low_d_trajectory_list.append(trajectory)
            colour_list.append('g')

        for trajectory in condition_2_low_d_trajectories:
            low_d_trajectory_list.append(trajectory)
            colour_list.append('r')

    # Perform JPCA
    low_d_trajectory_list = np.array(low_d_trajectory_list)
    jpca = jPCA.JPCA(num_jpcs=4)
    (projected, full_data_var, pca_var_capt, jpca_var_capt) = jpca.fit(low_d_trajectory_list)

    number_of_trajectories = np.shape(low_d_trajectory_list)[0]

    # Plot Trajectories
    for trajectory_index in range(number_of_trajectories):
        trajectory = projected[trajectory_index]
        colour = colour_list[trajectory_index]
        trajectory_axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=colour, alpha=0.1)

    # Scatter Start Points
    for trajectory_index in range(number_of_trajectories):
        trajectory = projected[trajectory_index]
        trajectory_axis.scatter([trajectory[0, 0]], [trajectory[0, 1], trajectory[0, 2]], c='k', alpha=0.1)

    trajectory_axis.axis('off')



# Load Data
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

# Create Model
number_of_sessions = len(session_list)
condition_names = None
trial_start = -6
trial_stop = 18
latent_dimensions = 64
number_of_factors = 25

input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length, condition_1_data_list, condition_2_data_list = load_data(session_list, condition_names, trial_start, trial_stop)
model = LFADS_Model_V3.LFADS_Model(session_neuron_numbers, session_trial_numbers, number_of_factors, trial_length, number_of_sessions, latent_dimensions)

# Load Model Weights
weights_file = "/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Model_Weights/7500_Model_weights"
model.load_weights(weights_file)

# Create Figure
figure_1 = plt.figure()
rows = 1
columns = 1
trajectory_axis = figure_1.add_subplot(rows, columns, 1, projection='3d')

# View Infered Trajectoreis
view_trajectories_jpca_interactive(model, number_of_sessions, condition_1_data_list, condition_2_data_list, trajectory_axis)
plt.show()