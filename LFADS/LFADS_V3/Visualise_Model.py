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

import Custom_Layers
import jPCA



def view_encoding(model, test_data, initial_conditions, save_directory, index):

    # Get Number Of Samples
    number_of_samples = np.shape(test_data)[0]

    # Create Colourmap
    colourmap = cm.get_cmap('gist_rainbow')

    # Get Encoder From Model
    encoder = model.encoder

    # Predict Data
    predicted_outputs = encoder(test_data)[2]

    for sample in range(number_of_samples):

        # Get Sample Colour
        colour_value = float(sample) / number_of_samples
        colour = colourmap(colour_value)

        # Get Points
        actual_point = initial_conditions[sample]
        predicted_point = predicted_outputs[sample]

        # Plot These Points
        plt.scatter([actual_point[0]], [actual_point[1]], c=np.array([colour]))
        plt.scatter([predicted_point[0]], [predicted_point[1]], c=np.array([colour]))
        plt.plot([actual_point[0], predicted_point[0]], [actual_point[1], predicted_point[1]], c=np.array([colour]))

    plt.savefig(save_directory + "\\" + str(index).zfill(4) + ".png")
    plt.close()



def view_trajectories(model, number_of_sessions, high_dimensional_data, trajectory_axis, epoch, divisor):

    low_d_trajectory_list = []

    # Get Encoder and Generator
    encoder = model.encoder
    generator = model.generator

    colourmap = cm.get_cmap('hsv')

    for session_index in range(number_of_sessions):

        # Input Translation Weight
        session_data = high_dimensional_data[session_index]
        #print("Session Data Shape", np.shape(session_data))

        # Translate To Factors
        session_input_translation_weights = model.input_translation_weights_list[session_index]
        session_input_translation_weights = np.array(session_input_translation_weights)
        session_data_factors = np.matmul(session_data, session_input_translation_weights)
        #print("Session Data Factors", np.shape(session_data_factors))

        # Encoder To Initial States
        z_mean, z_log_var, initial_states = encoder(session_data_factors)
        #print("Initial States", np.shape(initial_states))

        # Generate Low D Trajectories From Initial States
        low_d_trajectories = generator(initial_states)
        low_d_trajectory_list.append(low_d_trajectories)

        colour = colourmap(float(session_index)/number_of_sessions)
        for trajectory in low_d_trajectories:
            trajectory_axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.2, c=colour)

    angle = (float(epoch)/divisor) % 360
    trajectory_axis.view_init(30, angle)
    trajectory_axis.axis('off')
    trajectory_axis.set_title('Epoch: ' + str(epoch))

    return low_d_trajectory_list


def view_trajectories_jpca(model, number_of_sessions, condition_1_data_list, condition_2_data_list, trajectory_axis, epoch, divisor):


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

        # If We Have Condition 1 Type Data
        if np.shape(condition_1_data)[0] > 0:
            condition_1_data_factors = np.matmul(condition_1_data, session_input_translation_weights)
            condition_1_z_mean, condition_1_z_log_var, condition_1_initial_states = encoder(condition_1_data_factors)
            condition_1_low_d_trajectories = generator(condition_1_initial_states)

            for trajectory in condition_1_low_d_trajectories:
                low_d_trajectory_list.append(trajectory)
                colour_list.append('g')

        # If We Have Condition 2 Type Data
        if np.shape(condition_2_data)[0] > 0:
            condition_2_data_factors = np.matmul(condition_2_data, session_input_translation_weights)
            condition_2_z_mean, condition_2_z_log_var, condition_2_initial_states = encoder(condition_2_data_factors)
            condition_2_low_d_trajectories = generator(condition_2_initial_states)

            for trajectory in condition_2_low_d_trajectories:
                low_d_trajectory_list.append(trajectory)
                colour_list.append('r')


    # Perform JPCA
    low_d_trajectory_list = np.array(low_d_trajectory_list)
    jpca = jPCA.JPCA(num_jpcs=2)
    (projected, full_data_var, pca_var_capt, jpca_var_capt) = jpca.fit(low_d_trajectory_list)

    number_of_trajectories = np.shape(low_d_trajectory_list)[0]

    # Plot Trajectories
    for trajectory_index in range(number_of_trajectories):
        trajectory = projected[trajectory_index]
        colour = colour_list[trajectory_index]
        trajectory_axis.plot(trajectory[:, 0], trajectory[:, 1], c=colour, alpha=0.2)

    # Scatter Start Points
    for trajectory_index in range(number_of_trajectories):
        trajectory = projected[trajectory_index]
        colour = colour_list[trajectory_index]
        trajectory_axis.scatter([trajectory[0, 0]], [trajectory[0, 1]], c=colour, alpha=0.2)

    trajectory_axis.axis('off')
    trajectory_axis.set_title('Epoch: ' + str(epoch))




def view_trajectories_jpca_many(model, number_of_sessions, condition_1_data_list, condition_2_data_list, condition_3_data_list, condition_4_data_list, trajectory_axis, epoch, divisor):


    # Get Encoder and Generator
    encoder = model.encoder
    generator = model.generator

    colourmap = cm.get_cmap('hsv')

    low_d_trajectory_list = []
    colour_list = []
    alpha_list = []

    for session_index in range(number_of_sessions):

        # Input Translation Weight
        condition_1_data = condition_1_data_list[session_index]
        condition_2_data = condition_2_data_list[session_index]
        condition_3_data = condition_3_data_list[session_index]
        condition_4_data = condition_4_data_list[session_index]

        # Translate To Factors
        session_input_translation_weights = model.input_translation_weights_list[session_index]
        session_input_translation_weights = np.array(session_input_translation_weights)

        # If We Have Condition 1 Type Data
        if np.shape(condition_1_data)[0] > 0:
            condition_1_data_factors = np.matmul(condition_1_data, session_input_translation_weights)
            condition_1_z_mean, condition_1_z_log_var, condition_1_initial_states = encoder(condition_1_data_factors)
            condition_1_low_d_trajectories = generator(condition_1_initial_states)

            for trajectory in condition_1_low_d_trajectories:
                low_d_trajectory_list.append(trajectory)
                colour_list.append('m')
                alpha_list.append(0.8)

        # If We Have Condition 2 Type Data
        if np.shape(condition_2_data)[0] > 0:
            condition_2_data_factors = np.matmul(condition_2_data, session_input_translation_weights)
            condition_2_z_mean, condition_2_z_log_var, condition_2_initial_states = encoder(condition_2_data_factors)
            condition_2_low_d_trajectories = generator(condition_2_initial_states)

            for trajectory in condition_2_low_d_trajectories:
                low_d_trajectory_list.append(trajectory)
                colour_list.append('orange')
                alpha_list.append(0.8)

        # If We Have Condition 3 Type Data
        if np.shape(condition_3_data)[0] > 0:
            condition_3_data_factors = np.matmul(condition_3_data, session_input_translation_weights)
            condition_3_z_mean, condition_3_z_log_var, condition_3_initial_states = encoder(condition_3_data_factors)
            condition_3_low_d_trajectories = generator(condition_3_initial_states)

            for trajectory in condition_3_low_d_trajectories:
                low_d_trajectory_list.append(trajectory)
                colour_list.append('g')
                alpha_list.append(0.1)

        # If We Have Condition 4 Type Data
        if np.shape(condition_4_data)[0] > 0:
            condition_4_data_factors = np.matmul(condition_4_data, session_input_translation_weights)
            condition_4_z_mean, condition_4_z_log_var, condition_4_initial_states = encoder(condition_4_data_factors)
            condition_4_low_d_trajectories = generator(condition_4_initial_states)

            for trajectory in condition_4_low_d_trajectories:
                low_d_trajectory_list.append(trajectory)
                colour_list.append('b')
                alpha_list.append(0.1)

    # Perform JPCA
    low_d_trajectory_list = np.array(low_d_trajectory_list)
    jpca = jPCA.JPCA(num_jpcs=2)
    (projected, full_data_var, pca_var_capt, jpca_var_capt) = jpca.fit(low_d_trajectory_list)

    number_of_trajectories = np.shape(low_d_trajectory_list)[0]

    # Plot Trajectories
    for trajectory_index in range(number_of_trajectories):
        trajectory = projected[trajectory_index]
        colour = colour_list[trajectory_index]
        alpha = alpha_list[trajectory_index]
        trajectory_axis.plot(trajectory[:, 0], trajectory[:, 1], c=colour, alpha=alpha)

    # Scatter Start Points
    for trajectory_index in range(number_of_trajectories):
        trajectory = projected[trajectory_index]
        colour = colour_list[trajectory_index]
        alpha = alpha_list[trajectory_index]
        trajectory_axis.scatter([trajectory[0, 0]], [trajectory[0, 1]], c=colour, alpha=alpha)

    trajectory_axis.axis('off')
    trajectory_axis.set_title('Epoch: ' + str(epoch))


def view_trajectories_jpca_interactive(model, number_of_sessions, condition_1_data_list, condition_2_data_list, trajectory_axis, epoch, divisor):


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
        trajectory_axis.scatter([trajectory[0, 0]], [trajectory[0, 1], trajectory[0,2]], c='k', alpha=0.1)

    trajectory_axis.axis('off')
    trajectory_axis.set_title('Epoch: ' + str(epoch))


def view_decoding(model, low_dimensional_trajectories, high_dimensional_data, index, decoding_axis_list):

    # Get Model Layers and Weights
    decoding_weights = model.decoder_weights
    timeseries_activation_layer = model.timeseries_activation_layer

    # Reconstruct Rasters
    reconstructed_rasters = timeseries_activation_layer([low_dimensional_trajectories, decoding_weights])

    # Plot Reconstructions
    figure_1 = plt.figure()
    rows = 4
    columns = 2
    for sample in range(0, 6, 2):
        axis_1 = decoding_axis_list[sample]
        axis_2 = decoding_axis_list[sample + 1]

        axis_1.imshow(np.transpose(high_dimensional_data[sample]))
        axis_2.imshow(np.transpose(reconstructed_rasters[sample]))
        axis_2.axis('off')
        axis_1.axis('off')

        axis_1.set_title("Sample: " + str(sample) + " Actual")
        axis_2.set_title("Sample: " + str(sample) + " Predicted")

    #plt.savefig(save_directory + "/" + str(index).zfill(4) + ".png")
    #plt.close()



def visualise_model_lorentz(model, data_list, epoch_step, divisor, save_directory):

    number_of_sessions = len(data_list)

    # Create Figure
    figure_1 = plt.figure(constrained_layout=True, figsize=(14, 6))
    trajectory_axis = figure_1.add_subplot(1, 1, 1, projection='3d')

    # View Trajectories
    low_d_trajectory_list = view_trajectories(model, number_of_sessions, data_list, trajectory_axis, epoch_step, divisor)

    # Save Figure
    plt.draw()
    figure_1.savefig(os.path.join(save_directory, str(epoch_step).zfill(4) + ".png"))
    plt.close()

    np.save(os.path.join(save_directory, str(epoch_step).zfill(4) + ".npy"), low_d_trajectory_list)



def visualise_model(model, condition_1_data_list, condition_2_data_list, epoch_step, divisor, save_directory):

    # Get Number Of Sessions
    number_of_sessions = len(condition_1_data_list)

    # Create Figure
    figure_1 = plt.figure(constrained_layout=True, figsize=(14, 6))

    """
    figure_1_spec = gridspec.GridSpec(ncols=4, nrows=3, figure=figure_1)
    trajectory_axis             = figure_1.add_subplot(figure_1_spec[0:3, 0:2], projection='3d')
    decoding_1_real_axis        = figure_1.add_subplot(figure_1_spec[0, 2])
    decoding_1_predicted_axis   = figure_1.add_subplot(figure_1_spec[0, 3])
    decoding_2_real_axis        = figure_1.add_subplot(figure_1_spec[1, 2])
    decoding_2_predicted_axis   = figure_1.add_subplot(figure_1_spec[1, 3])
    decoding_3_real_axis        = figure_1.add_subplot(figure_1_spec[2, 2])
    decoding_3_predicted_axis   = figure_1.add_subplot(figure_1_spec[2, 3])
    """
    trajectory_axis = figure_1.add_subplot(1,1,1, projection='3d')

    """
    decoding_axis_list = [decoding_1_real_axis,
                          decoding_1_predicted_axis,
                          decoding_2_real_axis,
                          decoding_2_predicted_axis,
                          decoding_3_real_axis,
                          decoding_3_predicted_axis]
    """


    # View Trajectories
    view_trajectories_jpca(model, number_of_sessions, condition_1_data_list, condition_2_data_list, trajectory_axis, epoch_step, divisor)

    # View Decoded Rasters
    #view_decoding(model, low_dimensional_data, high_dimensional_data, epoch, decoding_axis_list)

    # Save Figures
    #plt.show()

    plt.draw()
    figure_1.savefig(os.path.join(save_directory, str(epoch_step).zfill(4) + ".png"))
    plt.close()





def visualise_model_many_conditions(model, condition_1_data_list, condition_2_data_list, condition_3_data_list, condition_4_data_list, epoch_step, divisor, save_directory):

    # Get Number Of Sessions
    number_of_sessions = len(condition_1_data_list)

    # Create Figure
    figure_1 = plt.figure(constrained_layout=True, figsize=(14, 6))
    trajectory_axis = figure_1.add_subplot(1,1,1, projection='3d')

    # View Trajectories
    view_trajectories_jpca_many(model, number_of_sessions, condition_1_data_list, condition_2_data_list, condition_3_data_list, condition_4_data_list, trajectory_axis, epoch_step, divisor)

    plt.draw()
    figure_1.savefig(os.path.join(save_directory, str(epoch_step).zfill(4) + ".png"))
    plt.close()
