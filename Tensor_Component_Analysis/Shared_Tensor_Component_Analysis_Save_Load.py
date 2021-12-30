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






def get_factor_contribution(neuron_factor, trial_factor, session_factor):

    trial_length = tf.shape(trial_factor)[1]
    number_of_neurons = tf.shape(neuron_factor)[1]
    number_of_trials = tf.shape(session_factor)[1]

    print("Trial Length", trial_length)
    print("Number of neurons", number_of_neurons)
    print("number of trials", number_of_trials)

    # Broadcast Neuron Factor To Entire Trial Length
    neuron_factor = tf.reshape(neuron_factor, [number_of_neurons, 1])
    factor_tensor = tf.broadcast_to(neuron_factor, [number_of_neurons, trial_length])

    # Multiply by trial factor along trial axis
    factor_tensor = tf.multiply(factor_tensor, trial_factor)

    # Broadcast Neuron Factor To Session Length
    factor_tensor = tf.expand_dims(factor_tensor, axis=-1)
    factor_tensor = tf.repeat(factor_tensor, number_of_trials, 2)

    # Multiply by Session Factor Along Session Axis
    factor_tensor = tf.multiply(factor_tensor, session_factor)

    return factor_tensor





def reconstruct_matrix(neuron_factors, trial_factors, session_factors):

    # Get Shape
    number_of_factors = np.shape(neuron_factors)[0]
    number_of_neurons = np.shape(neuron_factors)[1]
    trial_length = np.shape(trial_factors)[1]
    number_of_trials = np.shape(session_factors)[1]

    print("Neuron factors shape", tf.shape(neuron_factors))
    print("Number of factors", number_of_factors)

    # Create Empty Matrix To Hold Output
    reconstructed_matrix = tf.zeros([number_of_neurons, trial_length, number_of_trials])

    # Iterate Through Each Factor and Add Its Contribution To The Whole Matrix
    for factor in range(number_of_factors):
        neuron_factor = tf.slice(neuron_factors,   begin=[factor, 0], size=[1, number_of_neurons])
        trial_factor = tf.slice(trial_factors,     begin=[factor, 0], size=[1, trial_length])
        session_factor = tf.slice(session_factors, begin=[factor, 0], size=[1, number_of_trials])
        factor_tensor = get_factor_contribution(neuron_factor, trial_factor, session_factor)

        reconstructed_matrix = tf.add(reconstructed_matrix, factor_tensor)

    # Return Reconstruced Matrix
    return reconstructed_matrix



class tensor_decomposition_model(keras.Model):


    def __init__(self,  number_of_timepoints, number_of_factors, number_of_neurons, number_of_trials, **kwargs):
        super(tensor_decomposition_model, self).__init__(**kwargs)

        # Setup Variables
        self.trial_length      = number_of_timepoints
        self.number_of_factors = number_of_factors
        self.number_of_neurons = number_of_neurons
        self.number_of_trials  = number_of_trials

        # Create Shared Trial Weights
        self.trial_weights =   self.add_weight(shape=(self.number_of_factors, self.trial_length),       initializer='normal', trainable=True, name='trial_weights', constraint=tf.keras.constraints.NonNeg())
        self.neuron_weights =  self.add_weight(shape=(self.number_of_factors, self.number_of_neurons),  initializer='normal', trainable=True, name='neuron_weights', constraint=tf.keras.constraints.NonNeg())
        self.session_weights = self.add_weight(shape=(self.number_of_factors, self.number_of_trials),   initializer='normal', trainable=True, name='session_weights', constraint=tf.keras.constraints.NonNeg())

        # Setup Loss Tracking
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")


    def get_factor_contribution(self, neuron_factor, trial_factor, session_factor):

        # Broadcast Neuron Factor To Entire Trial Length
        neuron_factor = tf.reshape(neuron_factor, [self.number_of_neurons, 1])
        factor_tensor = tf.broadcast_to(neuron_factor, [self.number_of_neurons, self.trial_length])

        # Multiply by trial factor along trial axis
        factor_tensor = tf.multiply(factor_tensor, trial_factor)

        # Broadcast Neuron Factor To Session Length
        factor_tensor = tf.expand_dims(factor_tensor, axis=-1)
        factor_tensor = tf.repeat(factor_tensor, self.number_of_trials, 2)

        # Multiply by Session Factor Along Session Axis
        factor_tensor = tf.multiply(factor_tensor, session_factor)

        return factor_tensor

    def reconstruct_matrix(self, neuron_factors, trial_factors, session_factors):

        # Create Empty Matrix To Hold Output
        reconstructed_matrix = tf.zeros([self.number_of_neurons, self.trial_length, self.number_of_trials])

        # Iterate Through Each Factor and Add Its Contribution To The Whole Matrix
        for factor in range(number_of_factors):
            neuron_factor  = tf.slice(neuron_factors,  begin=[factor, 0], size=[1, self.number_of_neurons])
            trial_factor   = tf.slice(trial_factors,   begin=[factor, 0], size=[1, self.trial_length])
            session_factor = tf.slice(session_factors, begin=[factor, 0], size=[1, self.number_of_trials])
            factor_tensor = self.get_factor_contribution(neuron_factor, trial_factor, session_factor)

            reconstructed_matrix = tf.add(reconstructed_matrix, factor_tensor)

        # Return Reconstruced Matrix
        return reconstructed_matrix

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker]

    def call(self, data):

        reconstructed_matirices = []

        for matrix in data:
            reconstruction = self.reconstruct_matrix(self.neuron_weights, self.trial_weights, self.session_weights)
            reconstructed_matirices.append(reconstruction)

        return reconstructed_matirices

    def train_step(self, data):

        with tf.GradientTape() as tape:

            total_loss = 0
            for matrix in data:

                # Reconstruct Matrix
                reconstruction = self.reconstruct_matrix(self.neuron_weights, self.trial_weights, self.session_weights)

                # Create loss Functions
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(matrix, reconstruction), axis=-1))
                print("Reconstruction loss", reconstruction_loss)

                total_loss += reconstruction_loss

        # Get Gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update Loss
        self.reconstruction_loss_tracker.update_state(total_loss)

        return {"reconstruction_loss":  self.reconstruction_loss_tracker.result() }





def train_model(trial_tensors_list):

    number_of_sessions = len(trial_tensors_list)
    number_of_factors = 7
    model_list = []

    shared_trial_weights = tf.zeros((number_of_factors, trial_length))

    # Create The Models
    for session in range(number_of_sessions):
        number_of_neurons = np.shape(trial_tensors_list[session])[0]
        number_of_timepoints = np.shape(trial_tensors_list[session])[1]
        number_of_trials = np.shape(trial_tensors_list[session])[2]

        print("Number of neurons", number_of_neurons)
        print("Number of timepoints", number_of_timepoints)
        print("Number of trials", number_of_trials)

        model = tensor_decomposition_model(number_of_timepoints, number_of_factors, number_of_neurons, number_of_trials)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
        model_output = model([trial_tensors_list[session]])

        print("Output Shape", np.shape(model_output))
        model_list.append(model)

    # Train The Models
    for epoch in range(10):
        for session_index in range(number_of_sessions):

            # Select Model
            model = model_list[session_index]

            # Set Weights To Be Shared Trial Weights
            model.trial_weights = shared_trial_weights

            # Fit Data
            training_data = [trial_tensors_list[session_index]]
            training_data = np.array(training_data)
            print("Training Data Shape", np.shape(training_data))
            model(training_data)

            # Save Weights
            shared_trial_weights = model.trial_weights

number_of_factors = 7
trial_length = 10

session_1_neurons = 36
session_2_neurons = 34

session_1_trials = 104
session_2_trials = 102

session_1_tensor = np.random.uniform(low=0, high=1, size=(session_1_neurons, trial_length, session_1_trials))
session_2_tensor = np.random.uniform(low=0, high=1, size=(session_2_neurons, trial_length, session_2_trials))

combined_tensors = [session_1_tensor, session_2_tensor]

train_model(combined_tensors)

