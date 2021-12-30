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



def train_model(model, input_tensor, input_data, plot_save_directory, weight_save_directory, visualise=False):

    # Convered and Epoch Count Variables
    converged = False
    epoch_count = 0
    divisor = 100

    # Learning Rate Parameters
    initital_learning_rate = 0.01
    learning_rate_stop     = 0.00001
    current_learning_rate = initital_learning_rate
    learning_rate_decay_factor = 0.95

    monitoring_window_size = 6
    monitoring_window = []

    # KL Scale Parameters
    kl_start_step = 0
    steps_to_increase_kl_loss_over = 900
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
            Visualise_Model.visualise_model(model, input_data, epoch_count, divisor, plot_save_directory)
            model.save_weights(model_save_directory)

        # Check For Convergence
        if total_loss < loss_stop:
            print("Converged! ")
            converged = True
            Visualise_Model.visualise_model(model, input_data, epoch_count, divisor, plot_save_directory)

            # Save Model
            model.save_weights(model_save_directory)

        # Increment Epoch Count
        epoch_count += 1







os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Load Data
session_list = [r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_0",
                r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_1",
                r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_2",
                r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_3",
                r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_4"]



# Load Data
number_of_sessions = len(session_list)
input_tensor, input_data, session_neuron_numbers, session_trial_numbers, trial_length = load_data(session_list)
print("Data Shape", input_tensor.shape)
print("Session neuron numbers", session_neuron_numbers)
print("Session trial numbers", session_trial_numbers)

# Create Model
number_of_factors = 30
latent_dimensions = 3
model = LFADS_Model_V3.LFADS_Model(session_neuron_numbers, session_trial_numbers, number_of_factors, trial_length, number_of_sessions, latent_dimensions)

# Setup Save Directories
plot_save_directory = r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Output_Plots"
weight_save_directory = r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Model_Weights"

train_model(model, input_tensor, input_data, plot_save_directory, weight_save_directory, visualise=True)
