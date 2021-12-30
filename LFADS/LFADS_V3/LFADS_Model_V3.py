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

import Custom_Layers


class LFADS_Model(keras.Model):

    def __init__(self, number_of_neurons_list, session_trial_list, number_of_factors, number_of_timepoints, number_of_sessions, latent_dimensions, **kwargs):
        super(LFADS_Model, self).__init__(**kwargs)

        # Setup Variables
        self.number_of_neurons_list = number_of_neurons_list
        self.session_trial_list     = session_trial_list
        self.number_of_factors      = number_of_factors
        self.number_of_timepoints   = number_of_timepoints
        self.number_of_sessions     = number_of_sessions
        self.latent_dimensions      = latent_dimensions
        self.kl_scale               = 0

        # Create Shared Factor To Latent Space Encoding and Decoding Weights
        self.decoder_weights = self.add_weight(shape=(self.latent_dimensions, self.number_of_factors), initializer='normal', trainable=True, name='Decoder_Weights')

        # Create Session Dependent Translation Weights
        self.input_translation_weights_list = []
        self.output_translation_weights_list = []

        for session in range(number_of_sessions):

            session_neurons = self.number_of_neurons_list[session]

            input_translation_weights = tf.Variable(initial_value=tf.random.normal([session_neurons, self.number_of_factors]), trainable=True, name=str(session) + "_input_translation_weights")
            output_translation_weights = tf.Variable(initial_value=tf.random.normal([self.number_of_factors, session_neurons]), trainable=True, name=str(session) + "_output_translation_weights")

            self.input_translation_weights_list.append(input_translation_weights)
            self.output_translation_weights_list.append(output_translation_weights)


        # Create Model Components
        self.timeseries_activation_layer = Custom_Layers.timeseries_multiplication_layer(self.number_of_timepoints)
        self.encoder = self.create_encoder()
        self.generator = self.create_generator()

        print(self.encoder.summary())
        print(self.generator.summary())


        # Setup Loss Tracking
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    def create_encoder(self):

        # Create Encoder
        encoder_input_layer = Input(shape=(self.number_of_timepoints, self.number_of_factors))
        embedding_recurrent_layer_forward = GRU(128, name='embedding_recurrent_layer_forwards', go_backwards=False, recurrent_dropout=0.05)(encoder_input_layer)
        embedding_recurrent_layer_backwards = GRU(128, name='embedding_recurrent_layer_backwards', go_backwards=True, recurrent_dropout=0.05)(encoder_input_layer)
        embedding_concatenate_layer = layers.Concatenate()([embedding_recurrent_layer_forward, embedding_recurrent_layer_backwards])
        z_mean_layer = layers.Dense(self.latent_dimensions, name='z_mean_layer')(embedding_concatenate_layer)
        z_std_layer = layers.Dense(self.latent_dimensions, name='z_std_layer')(embedding_concatenate_layer)
        z_sampling_layer = Custom_Layers.sampling_layer()([z_mean_layer, z_std_layer])
        encoder = keras.Model(encoder_input_layer, [z_mean_layer, z_std_layer, z_sampling_layer], name="encoder")

        return encoder


    def create_generator(self):

        # Create Generator
        generator_input_layer = Input(shape=(self.latent_dimensions), name='generator_input_layer')
        #generator_recurrent_layer = Custom_Layers.custom_rnn_layer(100, self.number_of_timepoints, self.latent_dimensions)(generator_input_layer)
        generator_recurrent_layer = Custom_Layers.custom_gru(self.latent_dimensions, 100, self.number_of_timepoints)(generator_input_layer)
        generator = keras.Model(generator_input_layer, generator_recurrent_layer, name="generator")

        return generator


    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]


    def call(self, data):

        # Create Lists To Hold Output Of Call
        reconstruction_list = []
        z_mean_list = []
        z_log_list = []

        # Iterate Through Each Session
        for session_index in range(self.number_of_sessions):

            # Get Session Dependent Translation Weights
            session_input_translation_weights = self.input_translation_weights_list[session_index]
            session_output_translation_weights = self.output_translation_weights_list[session_index]

            # Get Session Data
            session_data = data[0, session_index]
            #print("Session Data", session_data.shape)
            session_data = session_data.to_tensor()
            #print("Session Data", session_data.shape)

            # Translate Into Factors
            session_data_factors = self.timeseries_activation_layer([session_data, session_input_translation_weights])
            #print("Session Data Factors", session_data_factors.shape)

            # Encode Factors Activity Into Initial States
            z_mean, z_log_var, z = self.encoder(session_data_factors)
            #print("Latent State Encoded Data", z.shape)

            # Create Low Dimensional Trajectories From Initial States
            low_dimensional_trajectory = self.generator(z)
            #print("Low D Trajectory", low_dimensional_trajectory.shape)

            # Decode Factor Activity From Low Dimensional Trajectories
            factor_reconstruction = self.timeseries_activation_layer([low_dimensional_trajectory, self.decoder_weights])
            #print("Factor Reconstruction", factor_reconstruction.shape)

            # Reconstruct Neural Data From Factor Activation
            reconstruction = self.timeseries_activation_layer([factor_reconstruction, session_output_translation_weights])
            #print("Reconstruction Shape", reconstruction.shape)

            reconstruction_list.append(reconstruction)
            z_mean_list.append(z_mean)
            z_log_list.append(z_log_var)

        return reconstruction_list, z_mean_list, z_log_list


    def train_step(self, data):

        # Turn On Gradient Tape
        with tf.GradientTape() as tape:

            total_loss = 0
            reconstruction_loss = 0
            kl_loss = 0

            # Iterate Through Each Session
            for session_index in range(self.number_of_sessions):

                # Get Session Dependent Translation Weights
                session_input_translation_weights = self.input_translation_weights_list[session_index]
                session_output_translation_weights = self.output_translation_weights_list[session_index]

                # Get Session Data
                session_data = data[0, session_index]
                #print("Session Data", session_data.shape)
                session_data = session_data.to_tensor()
                #print("Session Data", session_data.shape)

                # Translate Into Factors
                session_data_factors = self.timeseries_activation_layer([session_data, session_input_translation_weights])
                #print("Session Data Factors", session_data_factors.shape)

                # Encode Factors Activity Into Initial States
                z_mean, z_log_var, z = self.encoder(session_data_factors)
                #print("Latent State Encoded Data", z.shape)

                # Create Low Dimensional Trajectories From Initial States
                low_dimensional_trajectory = self.generator(z)
                #print("Low D Trajectory", low_dimensional_trajectory.shape)

                # Decode Factor Activity From Low Dimensional Trajectories
                factor_reconstruction = self.timeseries_activation_layer([low_dimensional_trajectory, self.decoder_weights])
                #print("Factor Reconstruction", factor_reconstruction.shape)

                # Reconstruct Neural Data From Factor Activation
                reconstruction = self.timeseries_activation_layer([factor_reconstruction, session_output_translation_weights])
                #print("Reconstruction Shape", reconstruction.shape)

                # Get Losses
                session_reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(session_data, reconstruction), axis=-1))

                session_kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                session_kl_loss = tf.reduce_mean(tf.reduce_sum(session_kl_loss, axis=1))

                reconstruction_loss += session_reconstruction_loss
                kl_loss += session_kl_loss
                total_loss += session_reconstruction_loss + (self.kl_scale * session_kl_loss)


        grads = tape.gradient(total_loss, self.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 200.0)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {"loss":                 self.total_loss_tracker.result(),
                "reconstruction_loss":  self.reconstruction_loss_tracker.result(),
                "kl_loss":              self.kl_loss_tracker.result(), }



