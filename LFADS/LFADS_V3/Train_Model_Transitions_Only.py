import math
import os.path

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
import time

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LinearRegression, Ridge

import LFADS_Model_V3
import Visualise_Model
import Import_Preprocessed_Data




def align_sessions(base_directory_list, condition_1_tensor_list, condition_2_tensor_list, number_of_factors, display=False):


    print("Tensor Shape", np.shape(condition_1_tensor_list[0]))

    # Get Average Responses for Each Condition
    average_condition_1_tensors = []
    average_condition_2_tensors = []

    for tensor in condition_1_tensor_list:
        average_tensor = np.mean(tensor, axis=0)
        average_condition_1_tensors.append(average_tensor)

    for tensor in condition_2_tensor_list:
        average_tensor = np.mean(tensor, axis=0)
        average_condition_2_tensors.append(average_tensor)

    print("Average Tensor Shape", np.shape(average_condition_1_tensors[0]))

    # Stack Condition Tensors
    stacked_tensor_list = []
    number_of_sessions = len(condition_2_tensor_list)
    for session in range(number_of_sessions):
        stacked_tensor = np.vstack([average_condition_1_tensors[session], average_condition_2_tensors[session]])
        stacked_tensor_list.append(stacked_tensor)
    print("Stacked Tensor Shape", np.shape(stacked_tensor_list[0]))

    # Combine Stacked Tensors
    grand_tensor = np.hstack(stacked_tensor_list)
    print("Grand Tensor Shape", np.shape(grand_tensor))

    # Perform PCA On This Tensor
    pca_model = PCA(n_components=number_of_factors)
    pca_model.fit(grand_tensor)

    #pca_components = pca_model.components_
    pca_traces = pca_model.transform(grand_tensor)
    print("PCA Traces Shape", np.shape(pca_traces))

    # Regress PCA Traces Onto Average Neuron Traces
    regresion_matrix_list = []
    for session in range(number_of_sessions):
        session_data = stacked_tensor_list[session]
        regression_model = Ridge()
        print("session data shape", np.shape(session_data))
        regression_model.fit(X=session_data, y=pca_traces)
        regression_coefficients = regression_model.coef_
        regresion_matrix_list.append(regression_coefficients)


    print("Regression Coefficient Shape", np.shape(regresion_matrix_list[0]))
    if display == True:
        figure_1 = plt.figure()
        rows = number_of_sessions
        columns = number_of_factors
        axis_list = []
        item_count = 0
        for session_index in range(number_of_sessions):
            for factor_index in range(number_of_factors):

                # Load Factor
                factor = regresion_matrix_list[session_index][factor_index]

                # Create Image
                image = view_cluster_vector(base_directory_list[session_index], factor)

                # Get Image Range
                image_range = np.max(np.abs(image))

                # Create Axis
                axis_list.append(figure_1.add_subplot(rows, columns, item_count + 1))

                # Display Image On Axis
                axis_list[item_count].set_title("Session: " + str(session_index) + " Factor: " + str(factor_index))
                axis_list[item_count].imshow(image, cmap='bwr', vmin=-1*image_range, vmax=image_range)
                axis_list[item_count].axis('off')
                item_count += 1

        plt.show()
    return regresion_matrix_list


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




def load_session_data(matlab_file_location, session_name, behaviour_matrix_directory, trial_start, trial_stop):

    # Load Matalb Data
    data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

    # Extract Delta F Matrix
    delta_f_matrix = data_object.dF
    delta_f_matrix = np.nan_to_num(delta_f_matrix)
    delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
    delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(behaviour_matrix_directory, session_name + "_preprocessed_basic", "Behaviour_Matrix.npy"), allow_pickle=True)
    print("Behaviour matrix", np.shape(behaviour_matrix))

    # Get Selected Trials
    perfect_transition_vis_1 = []
    imperfect_transition_vis_1 = []

    number_of_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(1, number_of_trials):

        # Load Trial Data
        trial_data = behaviour_matrix[trial_index]

        # Is Trial In Visual Block
        trial_type = trial_data[1]
        if trial_type == 1 or trial_type == 2:

            # Is Trial First In Block
            first_in_block = trial_data[9]
            if first_in_block == 1:

                # Is Trial A Miss
                lick_response = trial_data[2]
                if lick_response == 0:

                    # Check If Perfect Transition (Following Trial Is Correct)
                    if behaviour_matrix[trial_index + 1][3] == 1:
                        perfect_transition_vis_1.append(trial_index)

                    else:
                        imperfect_transition_vis_1.append(trial_index)

    # Get Selected Onsets
    perfect_transition_vis_1_onsets = []
    imperfect_transition_vis_1_onsets = []

    for trial in perfect_transition_vis_1:
        perfect_transition_vis_1_onsets.append(behaviour_matrix[trial][12])

    for trial in imperfect_transition_vis_1:
        imperfect_transition_vis_1_onsets.append(behaviour_matrix[trial][12])

    # Combine Into All Onsets
    all_onsets = perfect_transition_vis_1_onsets + imperfect_transition_vis_1_onsets

    # Get Tensors
    full_tensor = create_trial_tensor(delta_f_matrix, all_onsets, trial_start, trial_stop)
    condition_1_tensor = create_trial_tensor(delta_f_matrix, perfect_transition_vis_1_onsets, trial_start, trial_stop)
    condition_2_tensor = create_trial_tensor(delta_f_matrix, imperfect_transition_vis_1_onsets, trial_start, trial_stop)

    return condition_1_tensor, condition_2_tensor, full_tensor



def load_data(session_list, condition_names, trial_start, trial_stop, behaviour_matrix_trajectory):

    # Get Data Structure
    input_data = []
    session_neuron_numbers = []
    session_trial_numbers = []
    trial_length = 0
    condition_1_tensor_list = []
    condition_2_tensor_list = []

    for session in session_list:

        # Get Session Name
        session_name = session.split('/')[-1]
        session_name = session_name.replace("_preprocessed_basic.mat", "")
        print("Performing Tensor Component Analysis for Session: ", session_name)

        # Load Session Data
        condition_1_tensor, condition_2_tensor, full_tensor = load_session_data(session, session_name, behaviour_matrix_directory, trial_start, trial_stop)

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



def train_model(model, input_tensor, condition_1_data_list, condition_2_data_list, plot_save_directory, weight_save_directory, visualise=False):

    # Convered and Epoch Count Variables
    converged = False
    epoch_count = 0
    divisor = 100

    # Learning Rate Parameters
    initital_learning_rate = 0.001
    learning_rate_stop     = 0.00001
    current_learning_rate = initital_learning_rate
    learning_rate_decay_factor = 0.95

    monitoring_window_size = 6
    monitoring_window = []

    # KL Scale Parameters
    kl_start_step = 0
    steps_to_increase_kl_loss_over = 2000
    steps_to_increase_kl_loss_over = steps_to_increase_kl_loss_over * number_of_sessions
    kl_increment = float(1) / steps_to_increase_kl_loss_over
    current_kl_scale = 0

    kl_loss_threshold = 1
    loss_stop = 0.1

    # Compile Model
    optimiser = keras.optimizers.Adam(learning_rate=initital_learning_rate)
    model.compile(optimizer=optimiser)

    # Save Model

    while not converged:

        model_save_directory = os.path.join(weight_save_directory, str(epoch_count) + "_Model_weights")

        # Fit Model
        history = model.fit(input_tensor, epochs=1, verbose=0)

        # Get Loss
        total_loss = history.history["loss"][0]

        # Update Learning Rate
        if all(i < total_loss for i in monitoring_window):
            current_learning_rate = current_learning_rate * learning_rate_decay_factor
            if current_learning_rate < learning_rate_stop:
                current_learning_rate = learning_rate_stop

        # Add Loss To Monitoring Window
        monitoring_window.append(total_loss)
        if len(monitoring_window) > monitoring_window_size:
            monitoring_window.pop(0)


        # Scale KL Loss
        if epoch_count > kl_start_step:
            current_kl_scale += kl_increment
            if current_kl_scale > 1:
                current_kl_scale = 1
        model.kl_scale = current_kl_scale

        print("Epoch:", np.around(epoch_count, 6), " Current Loss:", np.around(total_loss, 6), " Learning Rate", np.around(current_learning_rate, 6), " KL Scale", np.around(current_kl_scale, 6))
        if epoch_count % divisor == 0:

            # Visualise
            Visualise_Model.visualise_model(model, condition_1_data_list, condition_2_data_list, epoch_count, divisor, plot_save_directory)
            model.save_weights(model_save_directory)

        # Check For Convergence
        if current_learning_rate < learning_rate_stop:
            print("Converged! ")
            converged = True
            Visualise_Model.visualise_model(model, condition_1_data_list, condition_2_data_list, epoch_count, divisor, plot_save_directory)

            # Save Model
            model.save_weights(model_save_directory)

        # Increment Epoch Count
        epoch_count += 1



def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list






os.environ['CUDA_VISIBLE_DEVICES'] = '-1'





# Load Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/Best_switching_sessions_all_sites"
session_list = load_matlab_sessions(base_directory)

# Model Parameters
number_of_factors = 25
latent_dimensions = 7

# Load Data
number_of_sessions = len(session_list)
condition_names = None
trial_start = -6
trial_stop = 16
behaviour_matrix_directory = r"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Behaviour_Matricies"
input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length, condition_1_tensor_list, condition_2_tensor_list = load_data(session_list, condition_names, trial_start, trial_stop, behaviour_matrix_directory)

# Create Model
model = LFADS_Model_V3.LFADS_Model(session_neuron_numbers, session_trial_numbers, number_of_factors, trial_length, number_of_sessions, latent_dimensions)

# Setup Save Directories
plot_save_directory = r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Output_Plots"
weight_save_directory = r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Model_Weights"


train_model(model, input_tensor, condition_1_tensor_list, condition_2_tensor_list, plot_save_directory, weight_save_directory, visualise=True)
