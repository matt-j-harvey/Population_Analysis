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

        if number_of_samples == None:
            number_of_samples = 0

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

        #print("batch size", batch_size)
        #print("latent  dimensions", latent_dimensions)

        # Create a  Standard Normal Distribution With This Shape
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dimensions))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class timeseries_multiplication_layer(keras.layers.Layer):

    def __init__(self, number_of_timepoints):
        super(timeseries_multiplication_layer, self).__init__()

        # Get Structure
        self.number_of_timepoints = number_of_timepoints

    def call(self, inputs):

        # Unpack Inputs
        data = inputs[0]
        weights = inputs[1]

        # Create Empty Tensor To Hold Data
        output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        output_index = 0

        # Iterate Through Each Input
        for trial in data:
            trial_output_list = []

            for timepoint in range(self.number_of_timepoints):
                activation_tensor = tf.tensordot(trial[timepoint], weights, 1)
                trial_output_list.append(activation_tensor)

            trial_output = tf.convert_to_tensor(trial_output_list)
            output = output.write(output_index, trial_output)
            output_index += 1

        return output.stack()







class timeseries_multiplication_layer_interal_weights(keras.layers.Layer):

    def __init__(self, input_units, number_of_timepoints, number_of_units):
        super(timeseries_multiplication_layer_interal_weights, self).__init__()

        # Get Structure
        self.input_units = input_units
        self.number_of_units = number_of_units
        self.number_of_timepoints = number_of_timepoints

        # Create Weights and Biases
        self.internal_weights = self.add_weight(name='internal_weights', shape=(self.input_units, self.number_of_units), initializer='random_normal', trainable=True)
        #self.internal_biases  = self.add_weight(name='internal_biases', shape=(self.number_of_units), initializer='zeros', trainable=True)


    def call(self, inputs):

        # Create Empty Tensor To Hold Data
        output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        output_index = 0

        # Iterate Through Each Input
        for trial in inputs:
            trial_output_list = []

            for timepoint in range(self.number_of_timepoints):
                activation_tensor = tf.tensordot(trial[timepoint], self.internal_weights, 1)
                #activation_tensor = tf.add(activation_tensor, self.internal_biases)
                #activation_tensor = tf.nn.relu(activation_tensor)
                trial_output_list.append(activation_tensor)

            trial_output = tf.convert_to_tensor(trial_output_list)
            output = output.write(output_index, trial_output)
            output_index += 1

        return output.stack()




class custom_rnn_layer(keras.layers.Layer):

    def __init__(self, units, number_of_timepoints, number_of_dimensions):
        super(custom_rnn_layer, self).__init__()

        # Get Shape
        self.number_of_units = units
        self.number_of_timepoints = number_of_timepoints
        self.number_of_dimensions = number_of_dimensions

        # Create Weight Variables
        self.input_weights     = self.add_weight(name='input_weights',     shape=(self.number_of_dimensions, self.number_of_units),  initializer='random_normal', trainable=True)

        self.recurrent_weights = self.add_weight(name='recurrent_weights', shape=(self.number_of_units,      self.number_of_units),       initializer='random_normal', trainable=True)
        self.recurrent_biases  = self.add_weight(name='recurrent_biases', shape=(self.number_of_units), initializer='zeros', trainable=True)

        self.output_weights    = self.add_weight(name='output_weights',    shape=(self.number_of_units,      self.number_of_dimensions),  initializer='random_normal', trainable=True)
        self.output_biases     = self.add_weight(name='recurrent_biases', shape=(self.number_of_dimensions), initializer='zeros', trainable=True)


    def run_rnn(self, external_input, internal_state):

        # Get Input Activation
        input_activation = tf.tensordot(external_input, self.input_weights, 1)

        # Get Recurrent Activation
        internal_activation = tf.tensordot(internal_state, self.recurrent_weights, 1)
        internal_activation = tf.add(internal_activation, self.recurrent_biases)

        # Add These Values
        activation_values = tf.add(input_activation, internal_activation)

        # Put Internal Activation Values Through Tanh Activation Function
        activation_values = tf.tanh(activation_values)

        # Get Output Activation
        output = tf.tensordot(activation_values, self.output_weights, 1)
        output = tf.add(output, self.output_biases)

        # Put Output Activation Values Through Tanh Activation Function
        output = tf.tanh(output)

        return output, activation_values

    def call(self, inputs):

        # Create Empty Tensor To Hold Data
        output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        output_index = 0

        # Iterate Through Each Input
        for input in inputs:

            trial_output_list = []

            # Set Initial Conditions
            internal_state = tf.zeros(self.number_of_units)
            external_input = input
            trial_output_list.append(external_input)

            for timepoint in range(1, self.number_of_timepoints):
                external_input, internal_state = self.run_rnn(external_input, internal_state)
                trial_output_list.append(external_input)

            trial_output = tf.convert_to_tensor(trial_output_list)

            output = output.write(output_index, trial_output)
            output_index += 1

        return output.stack()


class custom_gru(keras.layers.Layer):

    def __init__(self, input_dimensions, hidden_size, number_of_timepoints, dtype=tf.float32):
        super(custom_gru, self).__init__()

        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        self.number_of_timepoints = number_of_timepoints

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wz = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), trainable=True, name='wz')
        self.Wr = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), trainable=True, name='wr')
        self.Wh = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), trainable=True, name='wh')

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Uz = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), trainable=True, name='uz')
        self.Ur = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), trainable=True, name='ur')
        self.Uh = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), trainable=True, name='uh')

        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), trainable=True, name='br')
        self.bz = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), trainable=True, name='bz')
        self.bh = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), trainable=True, name='bh')

        # Create Output Weights and Biases
        self.output_weights = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.input_dimensions), mean=0, stddev=0.01), trainable=True, name='generator_output_weights')
        self.output_biases =  tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.input_dimensions, ), mean=0, stddev=0.01), trainable=True, name='generator_output_biases')

        # Define the input layer placeholder


    def forward_pass(self, h_tm1, x_t):

        # 1.) Calculate Reset Vector
        r_t = tf.sigmoid(tf.tensordot(x_t, self.Wr, 1) + tf.tensordot(h_tm1, self.Ur, 1) + self.br)

        # 2.) Using the Reset Vector - Create a Candidate Hidden State
        proposal_input_component = tf.tensordot(x_t, self.Wh, 1)
        proposal_reset_component = tf.tensordot(tf.multiply(r_t, h_tm1), self.Uh, 1)
        h_proposal = proposal_input_component + proposal_reset_component
        h_proposal = h_proposal + self.bh
        h_proposal = tf.tanh(h_proposal)

        # 3.) Calculate The Update Vector
        z_t = tf.tensordot(x_t, self.Wz, 1) + tf.tensordot(h_tm1, self.Uz, 1)
        z_t = z_t + self.bz
        z_t = tf.sigmoid(z_t)

        # 4.) Using the Update Vector combine the previous hidden state and the candiate state in a certain ratio to create the final Hidden state
        h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

        # Clip
        h_t = tf.clip_by_value(h_t, clip_value_max=5, clip_value_min=-5)

        # 5.) Multiply Hidden State by Output Vectors To Get Output
        output = tf.tensordot(h_t, self.output_weights, 1)
        output = tf.add(output, self.output_biases)

        return h_t, output


    def call(self, inputs):

        # Create Empty Tensor To Hold Data
        output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        output_index = 0

        # Iterate Through Each Input
        for input in inputs:

            trial_output_list = []

            # Set Initial Conditions
            internal_state = tf.zeros(self.hidden_size)
            external_input = input
            trial_output_list.append(external_input)

            for timepoint in range(1, self.number_of_timepoints):
                internal_state, external_input = self.forward_pass(internal_state, external_input)
                trial_output_list.append(external_input)

            trial_output = tf.convert_to_tensor(trial_output_list)

            output = output.write(output_index, trial_output)
            output_index += 1

        return output.stack()

