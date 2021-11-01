import os.path

import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Input
from keras.layers import LSTM, Dense, GRU, ZeroPadding2D
from sklearn.model_selection import train_test_split
import random
from keras.callbacks import LearningRateScheduler
from sklearn.decomposition import TruncatedSVD, PCA, NMF, FactorAnalysis
from sklearn.linear_model import LinearRegression
import cv2
import jPCA

import sys
import LFADS_Model
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
        regression_model = LinearRegression()
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





def get_tensor_svds(tensor, factors):

    number_of_samples = np.shape(tensor)[0]
    number_of_timepoints =np.shape(tensor)[1]
    number_of_neurons = np.shape(tensor)[2]

    tensor = np.ndarray.reshape(tensor, (number_of_samples * number_of_timepoints, number_of_neurons))
    svd_model = TruncatedSVD(n_components=factors)
    svd_model.fit(tensor)
    return svd_model.components_


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


def load_data(matlab_file, condition_names, trial_start, trial_stop):

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



def plot_all_trajectories(model_list, weights_save_directory, condition_1_tensor_list, condition_2_tensor_list, plot_directory, batch_size, epoch):

    # Create Empty Lists
    condition_1_trajectories_list = []
    condition_2_trajectories_list = []

    # Iterate Through All Sessions
    number_of_sessions = len(model_list)
    for session_index in range(number_of_sessions):

        # Select Model
        model = model_list[session_index]

        # Load Weights
        model.load_weights(weights_save_directory, by_name=True)

        # Get Trajectories
        condition_1_trajectories = get_model_trajectories(model, condition_1_tensor_list[session_index], batch_size)
        condition_2_trajectories = get_model_trajectories(model, condition_2_tensor_list[session_index], batch_size)

        # Add To Liist
        condition_1_trajectories_list.append(condition_1_trajectories)
        condition_2_trajectories_list.append(condition_2_trajectories)

    # Plot These Trajectories
    latent_dimensions = np.shape(condition_2_trajectories_list[0])[2]

    # PLot Latent Space
    fig = plt.figure()
    if latent_dimensions > 2:
        fig.clf()
        ax = plt.axes(projection='3d')

        for condition_1_trajectories in condition_1_trajectories_list:
            for trajectory in condition_1_trajectories:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='b', alpha=0.2)
                ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]], c='navy', alpha=0.2)

        for condition_2_trajectories in condition_2_trajectories_list:
            for trajectory in condition_2_trajectories:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='g', alpha=0.2)
                ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]], c='darkgreen', alpha=0.2)


    elif latent_dimensions == 2:
        fig.clf()
        ax = plt.axes()

        for condition_1_trajectories in condition_1_trajectories_list:
            for trajectory in condition_1_trajectories:
                ax.plot(trajectory[:, 0], trajectory[:, 1], c='b', alpha=0.2)
                ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], c='navy', alpha=0.2)

        for condition_2_trajectories in condition_2_trajectories_list:
            for trajectory in condition_2_trajectories:
                ax.plot(trajectory[:, 0], trajectory[:, 1], c='g', alpha=0.2)
                ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], c='darkgreen', alpha=0.2)



    ax.set_title(str(epoch))
    plt.draw()
    plt.savefig(plot_directory + "//" + str(epoch).zfill(4) + ".png")
    plt.close()


def get_model_trajectories(model, data, batch_size):

    # Get Batch Sized Data
    number_of_samples    = np.shape(data)[0]
    number_of_timepoints = np.shape(data)[1]
    number_of_neurons    = np.shape(data)[2]

    # If Number Of Samples > Batch Size - No Worries!
    if number_of_samples > batch_size:
        batch_data = data[0:batch_size]
        translated_input = model.timeseries_activation_layer([batch_data, model.translation_weights])
        initial_positions = model.encoder(translated_input)[2]
        trajectories = model.generator(initial_positions)

    # Else Give THe Network A Padded Input Data
    else:
        batch_data = np.zeros((batch_size, number_of_timepoints, number_of_neurons))
        batch_data[0:number_of_samples] = data
        translated_input = model.timeseries_activation_layer([batch_data, model.translation_weights])
        initial_positions = model.encoder(translated_input)[2]
        trajectories = model.generator(initial_positions)
        trajectories = trajectories[0:number_of_samples]

    # Reshape Data
    """
    trajectories = np.array(trajectories)
    number_of_samples    = np.shape(trajectories)[0]
    number_of_timepoints = np.shape(trajectories)[1]
    number_of_dimensions = np.shape(trajectories)[2]
    trajectories = np.ndarray.reshape(trajectories, (number_of_samples * number_of_timepoints, number_of_dimensions))

    # Perform PCA On Trajectories
    pca_model = PCA(n_components=3)
    low_d_trajectories = pca_model.fit_transform(trajectories)

    # Reshape Back
    low_d_trajectories = np.ndarray.reshape(low_d_trajectories, (number_of_samples, number_of_timepoints, 3))
    """
    # Convert Tensors To List
    trial_list = []
    for trajectory in trajectories:
        print("Trajectory Shape", np.shape(trajectory))
        trial_list.append(trajectory)

    # Perform JPCA
    jpca = jPCA.JPCA(num_jpcs=4)
    (projected, full_data_var, pca_var_capt, jpca_var_capt) = jpca.fit(trajectories)
    print("Projected", np.shape(projected))

    return projected



def learning_rate_function(epoch, lr):

    global epoch_count
    global number_of_sessions

    epoch_count += float(1) / number_of_sessions
    step_number = np.floor(epoch_count / epoch_drop)
    scaling_factor = drop ** step_number
    learning_rate = initial_learning_rate * scaling_factor
    print("Epoch Count: ", epoch_count)
    print("Learning Rate: ", learning_rate)
    return learning_rate



def get_batch_data(batch_size, tensor):

    # Get Data Structure
    number_of_samples    = np.shape(tensor)[0]
    number_of_timepoints = np.shape(tensor)[1]
    number_of_neurons    = np.shape(tensor)[2]

    # Create Empty Array To Hold Batch Data
    batch_data = np.zeros((batch_size, number_of_timepoints, number_of_neurons))

    # Populate Batch Data Array
    for sample_index in range(batch_size):
        data_index = np.random.randint(low=0, high=number_of_samples-1)
        batch_data[sample_index] = tensor[data_index]

    return batch_data


def train_lfads_model(file_list, save_directory, conditions, trial_start, trial_stop):

    # Settings
    latent_dimensions = 3
    number_of_factors = 15
    number_of_timepoints = trial_stop - trial_start


    global epoch_count
    global epoch_drop
    global drop
    global initial_learning_rate
    global epoch_stop
    initial_learning_rate = 0.0001
    epoch_count = 0
    epoch_drop = 50
    drop = 0.95
    epoch_step = 1 #From 50


    #Create Directory To Save Plots
    plot_directory = save_directory + "/LFADS_Training_Plots/"
    if not os.path.exists(plot_directory):
        os.mkdir(plot_directory)

    # Set Weights Save Directory
    weights_save_directory = save_directory + "/Model_Weights.h5"

    # Load Data
    data_list = []
    sample_size_list = []
    neuron_size_list = []
    session_indexes_list = []
    condition_1_tensor_list = []
    condition_2_tensor_list = []

    for file in file_list:

        # Load data
        condition_1_tensor, condition_2_tensor, full_tensor = load_data(file, conditions, trial_start, trial_stop)
        condition_1_tensor_list.append(condition_1_tensor)
        condition_2_tensor_list.append(condition_2_tensor)
        data_list.append(full_tensor)


        # Get Data Structure
        tensor_1_samples = np.shape(condition_1_tensor)[0]
        tensor_2_samples = np.shape(condition_2_tensor)[0]
        number_of_neurons = np.shape(full_tensor)[2]
        session_indexes = list(range(tensor_1_samples + tensor_2_samples))
        sample_size_list.append(tensor_1_samples)
        sample_size_list.append(tensor_2_samples)
        neuron_size_list.append(number_of_neurons)
        session_indexes_list.append(session_indexes)


    batch_size = np.min(sample_size_list)
    batch_size = 256

    global number_of_sessions
    number_of_sessions = len(file_list)


    # Create Model
    model_list = []
    for session_index in range(number_of_sessions):
        model = LFADS_Model.LFADS_Model(neuron_size_list[session_index], number_of_timepoints, 6, number_of_factors, 'session_' + str(session_index), batch_size)
        model.compile(optimizer=keras.optimizers.Adam())
        model(data_list[session_index][0:batch_size])
        model_list.append(model)


    # Set Initial Translation Matricies
    translation_weights_list = align_sessions(file_list, condition_1_tensor_list, condition_2_tensor_list, number_of_factors)
    for session_index in range(number_of_sessions):
        model_list[session_index].translation_weights = np.transpose(translation_weights_list[session_index])


    # Save Initial Model Weights
    for session_index in range(number_of_sessions):
        model_list[session_index].save_weights(weights_save_directory, save_format='h5')


    # Train LFADS Models
    for epoch in range(1000):
        print("Epoch", epoch)

        for session_index in range(number_of_sessions):
            print("Session: ", session_index)

            # Select Model
            model = model_list[session_index]

            # Load Weights
            model.load_weights(weights_save_directory, by_name=True)

            # Get Training Data
            batch_data = get_batch_data(batch_size, data_list[session_index])

            # Fit Model
            model.fit(batch_data, epochs=epoch_step, batch_size=batch_size, callbacks=[LearningRateScheduler(learning_rate_function, verbose=1)])

            # Save Weights
            model.save_weights(weights_save_directory, save_format='h5')

            # Scale Model Loss Function
            model.kl_scale = model.kl_scale + 0.001
            if model.kl_scale > 1: model.kl_scale = 1
            print("KL Scale", model.kl_scale)

        plot_all_trajectories(model_list, weights_save_directory, condition_1_tensor_list, condition_2_tensor_list, plot_directory, batch_size, epoch)




"""
base_directory = "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/"
file_list = []
files = os.listdir(base_directory)
for file in files:
    file_list.append(base_directory + file)
"""
"""
file_list = ["/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201022_112044__ACV004_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201016_113151__ACV004_B2_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201026_103629__ACV014_B3_SWITCH_preprocessed_basic.mat",]
"""

file_list = ["/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201022_112044__ACV004_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201016_113151__ACV004_B2_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201026_103629__ACV014_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201024_104327__ACV005_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201026_122511__ACV011_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201103_160924__ACV013_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201029_145825__ACV011_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201021_121703__ACV005_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201027_140620__ACV013_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20200922_114059__ACV003_B3_SWITCH_preprocessed_basic.mat"]

#"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201017_124502__ACV007_B3_SWITCH_preprocessed_basic.mat"
#"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201020_154541__ACV010_B4_SWITCH_preprocessed_basic.mat"
#"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201021_171440__ACV008_B3_SWITCH_preprocessed_basic.mat"
#"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201020_172309__ACV009_B3_SWITCH_preprocessed_basic.mat"

save_directory = r"/media/matthew/29D46574463D2856/Nick_LFADS_Weights/"

# Set Trial Details
conditions = ["odour_context_stable_vis_1",
              "visual_context_stable_vis_1"]

trial_start = 0
trial_stop = 100

train_lfads_model(file_list, save_directory, conditions, trial_start, trial_stop)