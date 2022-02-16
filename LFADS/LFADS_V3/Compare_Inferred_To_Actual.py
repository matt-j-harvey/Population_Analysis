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


def load_data(session_list):

    # Get Data Structure
    input_data = []
    session_neuron_numbers = []
    session_trial_numbers = []
    trial_length = 0

    for session in session_list:

        # Load Session Data
        session_data = np.load(os.path.join(session, "High_Dimensional_Data.npy"))

        # Get Session Structure
        session_trials = np.shape(session_data)[0]
        trial_length = np.shape(session_data)[1]
        session_neurons = np.shape(session_data)[2]

        input_data.append(session_data)
        session_neuron_numbers.append(session_neurons)
        session_trial_numbers.append(session_trials)

        print("Session", session, "Number Of Trials", session_trials, "Number of neurons", session_neurons)

    input_tensor = [input_data]
    input_tensor = tf.ragged.constant(input_tensor, dtype=tf.float32)

    return input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length


def load_low_d_data(session_list):

    low_d_data = []

    for session in session_list:
        session_data = np.load(os.path.join(session, "Low_Dimensional_Data.npy"))
        low_d_data.append(session_data)

    return low_d_data


def plot_low_d_trajectories(low_d_data, trajectory_axis):

    colourmap = cm.get_cmap('hsv')

    for session_index in range(number_of_sessions):

        low_d_trajectories = low_d_data[session_index]
        colour = colourmap(float(session_index) / number_of_sessions)
        for trajectory in low_d_trajectories:
            trajectory_axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.2, c=colour)

    trajectory_axis.axis('off')



# Load Data

session_list = [r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_0",
                r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_1",
                r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_2",
                r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_3",
                r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_4"]


number_of_sessions = len(session_list)
input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length = load_data(session_list)
print("Data Shape", input_tensor.shape)
print("Session neuron numbers", session_neuron_numbers)
print("Session trial numbers", session_trial_numbers)

# Create Model
number_of_factors = 30
latent_dimensions = 3
model = LFADS_Model_V3.LFADS_Model(session_neuron_numbers, session_trial_numbers, number_of_factors, trial_length, number_of_sessions, latent_dimensions)


# Load Model Weights
weights_file = "/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Model_Weights/5000_Model_weights"
model.load_weights(weights_file)

# Load Low D Data
low_d_data = load_low_d_data(session_list)
print("Low D Data Shape", np.shape(low_d_data[0]))


# Create Figure
figure_1 = plt.figure()
rows = 1
columns = 2
inferred_axis = figure_1.add_subplot(rows, columns, 1, projection='3d')
real_axis = figure_1.add_subplot(rows, columns, 2, projection='3d')


# View Infered Trajectoreis
Visualise_Model.view_trajectories(model, number_of_sessions, input_data, inferred_axis, 1, 1)
inferred_axis.set_title("Inferred Trajectories")

# View Real Trajectories
plot_low_d_trajectories(low_d_data, real_axis)
real_axis.set_title("Real Trajectories")


plt.show()