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








class custom_padding_layer(keras.layers.Layer):

    def __init__(self, number_of_timepoints, number_of_latent_dimensions, **kwargs):
        self.number_of_timepoints = number_of_timepoints
        self.number_of_latent_dimensions = number_of_latent_dimensions
        super(custom_padding_layer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['number_of_timepoints'] = self.number_of_timepoints
        config['number_of_latent_dimensions'] = self.number_of_latent_dimensions
        return config

    def build(self, input_dim):
        pass

    def call(self, inputs):

        # Get Input Dimensions
        number_of_samples = inputs.get_shape()[0]
        zero_tensor = tf.zeros([number_of_samples, self.number_of_timepoints-1, self.number_of_latent_dimensions])
        inputs = tf.reshape(inputs, [number_of_samples, 1, self.number_of_latent_dimensions])
        output = tf.concat([inputs, zero_tensor], axis=1)

        return output



class sampling_layer(layers.Layer):

    def call(self, inputs):

        # Get Inputs
        z_mean, z_log_var = inputs

        # Get Input Dimensions
        batch_size        = tf.shape(z_mean)[0]
        latent_dimensions = tf.shape(z_mean)[1]

        # Create a  Standard Normal Distribution With This Shape
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dimensions))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class timeseries_multiplication_layer(keras.layers.Layer):

    def call(self, inputs):

        # Unpack Inputs
        timeseries_input = inputs[0]
        weight_matrix = inputs[1]

        # Multiply Inputs
        output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        output_index = 0
        for timeseries in timeseries_input:
            factor_activation_tensor = tf.matmul(timeseries, weight_matrix)
            output = output.write(output_index, factor_activation_tensor)
            output_index += 1

        return output.stack()



class timeseries_exponentiation_layer(keras.layers.Layer):

    def call(self, inputs):

        # Create Output Tensor To Hold Results
        output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        output_index = 0
        for timepoint in inputs:
            exponentiated_timepoint = tf.math.exp(timepoint)
            output = output.write(output_index, exponentiated_timepoint)
            output_index += 1

        return output.stack()



class LFADS_Model(keras.Model):

    def __init__(self, number_of_neurons, number_of_timepoints, latent_dimensions, number_of_factors, session_name, batch_size, **kwargs):
        super(LFADS_Model, self).__init__(**kwargs)

        # Setup Variables
        self.number_of_neurons = number_of_neurons
        self.number_of_timepoints = number_of_timepoints
        self.latent_dimensions = latent_dimensions
        self.session_name = session_name
        self.batch_size = batch_size
        self.number_of_factors = number_of_factors

        self.translation_input_weights = self.add_weight(name= str(session_name) + "_Translation_Input_Weights",
                                                   shape=(self.number_of_neurons, self.number_of_factors),
                                                   initializer='normal',
                                                   trainable=True)

        self.translation_output_weights = self.add_weight(name= str(session_name) + "_Translation_Output_Weights",
                                                   shape=(self.number_of_factors, self.number_of_neurons),
                                                   initializer='normal',
                                                   trainable=True)

        self.latent_space_mapping_weights = self.add_weight(name="Latent_Space_Mapping_Weights",
                                                   shape=(self.latent_dimensions, self.number_of_factors),
                                                   initializer='normal',
                                                   trainable=True)

        self.timeseries_activation_layer = timeseries_multiplication_layer()
        self.exponentiation_layer = timeseries_exponentiation_layer()
        self.kl_scale = 0.01

        # Create Model Components
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
        embedding_recurrent_layer_forward = GRU(128, name='embedding_recurrent_layer_forwards',    go_backwards=False)(encoder_input_layer)
        embedding_recurrent_layer_backwards = GRU(128, name='embedding_recurrent_layer_backwards', go_backwards=True)(encoder_input_layer)
        embedding_concatenate_layer = layers.Concatenate()([embedding_recurrent_layer_forward,embedding_recurrent_layer_backwards])
        z_mean_layer     = layers.Dense(self.latent_dimensions)(embedding_concatenate_layer)
        z_std_layer      = layers.Dense(self.latent_dimensions)(embedding_concatenate_layer)
        z_sampling_layer = sampling_layer()([z_mean_layer, z_std_layer])

        # Combine Into Model
        encoder = keras.Model(encoder_input_layer, [z_mean_layer, z_std_layer, z_sampling_layer], name="encoder")

        return encoder


    def create_generator(self):

        # Create Generator
        generator_input_layer = Input(shape=(self.latent_dimensions), name='generator_input_layer', batch_size=self.batch_size)
        generator_padding_layer = custom_padding_layer(self.number_of_timepoints, self.latent_dimensions)(generator_input_layer)
        generator_recurrent_layer = GRU(100, return_sequences=True, name='generator_recurrent_layer', activation='tanh',recurrent_activation='tanh', activity_regularizer=l2(0.0001), recurrent_regularizer=l2(0.01))(generator_padding_layer)
        generator_output_layer = Dense(self.latent_dimensions, name='generator_output_layer')(generator_recurrent_layer)
        generator = keras.Model(generator_input_layer, generator_output_layer, name="generator")
        return generator

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, data):

        # Encode Session Neural Data Into Factors
        factor_activity = self.timeseries_activation_layer([data, self.translation_input_weights])

        # Encode Factor Activity Into Initial States
        z_mean, z_log_var, z = self.encoder(factor_activity)

        # Create Low Dimensional Trajectories From Initial States
        low_dimensional_trajectories = self.generator(z)

        # Decode Factor Activity From Low Dimensional Trajectories
        reconstructed_factor_activity = self.timeseries_activation_layer([low_dimensional_trajectories, self.latent_space_mapping_weights])

        # Translation Factor Activity Back Into Neural Activity
        reconstruction = self.timeseries_activation_layer([reconstructed_factor_activity, self.translation_output_weights])
        #reconstruction = self.timeseries_activation_layer([reconstructed_factor_activity, tf.linalg.pinv(self.translation_input_weights)])

        # Exponentiate for some reason
        reconstruction = self.exponentiation_layer(reconstruction)

        return reconstruction

    def train_step(self, data):

        with tf.GradientTape() as tape:
            # Encode Session Neural Data Into Factors
            factor_activity = self.timeseries_activation_layer([data, self.translation_input_weights])

            # Encode Factor Activity Into Initial States
            z_mean, z_log_var, z = self.encoder(factor_activity)

            # Create Low Dimensional Trajectories From Initial States
            low_dimensional_trajectories = self.generator(z)

            # Decode Factor Activity From Low Dimensional Trajectories
            reconstructed_factor_activity = self.timeseries_activation_layer([low_dimensional_trajectories, self.latent_space_mapping_weights])

            # Translation Factor Activity Back Into Neural Activity
            reconstruction = self.timeseries_activation_layer([reconstructed_factor_activity, self.translation_output_weights])
            #reconstruction = self.timeseries_activation_layer([reconstructed_factor_activity, tf.linalg.pinv(self.translation_input_weights)])

            # Exponentiate for some reason
            reconstruction = self.exponentiation_layer(reconstruction)

            # Create loss Functions
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mean_squared_error(data, reconstruction), axis=-1))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #total_loss = reconstruction_loss + (self.kl_scale * kl_loss) #0.02 #0.1 pretty good
            total_loss = reconstruction_loss + (1 * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 200.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {"loss":                 self.total_loss_tracker.result(),
                "reconstruction_loss":  self.reconstruction_loss_tracker.result(),
                "kl_loss":              self.kl_loss_tracker.result(), }




