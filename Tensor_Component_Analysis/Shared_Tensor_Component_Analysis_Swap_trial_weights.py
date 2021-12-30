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

import Import_Preprocessed_Data




def align_sessions(condition_tensor_list, number_of_factors):


    print("Tensor Shape", np.shape(condition_1_tensor_list[0]))

    average_tensors_all_conditions = []

    # Get Average Responses for Each Condition For Each Session
    number_of_conditions = len(condition_tensor_list)
    number_of_sessions = np.shape(condition_tensor_list[0])

    for condition_index in range(number_of_conditions):
        condition_average_tensors = []

        for session_index in range(number_of_sessions):
            tensor = condition_tensor_list[condition_index][session_index]
            average_tensor = np.mean(tensor, axis=0)
            condition_average_tensors.append(average_tensor)

        average_tensors_all_conditions.append(condition_average_tensors)
    print("Average Tensor Shape", np.shape(average_tensors_all_conditions))


    # Stack Condition Tensors
    stacked_tensor_list = []
    for session_index in range(number_of_sessions):
        condition_tensor_list = []
        for condition_index in range(number_of_conditions):
            condition_tensor_list.append(average_tensors_all_conditions[condition_index][session_index])
        stacked_tensor = np.vstack(condition_tensor_list)
    stacked_tensor_list.append(stacked_tensor)
    print("Stacked Tensor Shape", np.shape(stacked_tensor_list))

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

    return regresion_matrix_list





def get_tensor_svds(tensor, factors):

    number_of_samples = np.shape(tensor)[0]
    number_of_timepoints =np.shape(tensor)[1]
    number_of_neurons = np.shape(tensor)[2]

    tensor = np.ndarray.reshape(tensor, (number_of_samples * number_of_timepoints, number_of_neurons))
    svd_model = TruncatedSVD(n_components=factors)
    svd_model.fit(tensor)
    return svd_model.components_







def smooth_delta_f_matrix(delta_f_maxtrix):

    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_delta_f = []

    for trace in delta_f_maxtrix:
        smoothed_trace = np.convolve(trace, kernel, mode='same')
        smoothed_delta_f.append(smoothed_trace)

    smoothed_delta_f = np.array(smoothed_delta_f)
    smoothed_delta_f = np.nan_to_num(smoothed_delta_f)
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

    # Remove Nans
    delta_f_matrix = np.nan_to_num(delta_f_matrix)

    # Put Back Into Original Shape
    delta_f_matrix = np.transpose(delta_f_matrix)

    return delta_f_matrix



def get_block_boundaries(combined_onsets, visual_context_onsets, odour_context_onsets):

    visual_blocks = []
    odour_blocks = []

    current_block_start = 0
    current_block_end = None

    # Get Initial Onset
    if combined_onsets[0] in visual_context_onsets:
        current_block_type = 0
    elif combined_onsets[0] in odour_context_onsets:
        current_block_type = 1
    else:
        print("Error! onsets not in either vidual or oflactory onsets")

    # Iterate Through All Subsequent Onsets
    number_of_onsets = len(combined_onsets)
    for onset_index in range(1, number_of_onsets):

        # Get Onset
        onset = combined_onsets[onset_index]

        # If we are currently in an Visual Block
        if current_block_type == 0:

            # If The Next Onset is An Odour Block - Block Finish, add Block To Boundaries
            if onset in odour_context_onsets:
                current_block_end = onset_index-1
                visual_blocks.append([current_block_start, current_block_end])
                current_block_type = 1
                current_block_start = onset_index

        # If we Are currently in an Odour BLock
        if current_block_type == 1:

            # If The NExt Onset Is a Visual Trial - BLock Finish Add Block To Block Boundaires
            if onset in visual_context_onsets:
                current_block_end = onset_index - 1
                odour_blocks.append([current_block_start, current_block_end])
                current_block_type = 0
                current_block_start = onset_index

    return visual_blocks, odour_blocks








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
    selected_data = np.swapaxes(selected_data, 0, 2)
    return selected_data



def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list


class tensor_decomposition_model(keras.Model):

    def __init__(self, number_of_neurons, number_of_timepoints, number_of_trials, number_of_factors, non_negative, **kwargs):
        super(tensor_decomposition_model, self).__init__(**kwargs)

        # Setup Variables
        self.number_of_neurons = number_of_neurons
        self.trial_length = number_of_timepoints
        self.number_of_trials = number_of_trials
        self.number_of_factors = number_of_factors

        # Create Weight Variables
        if non_negative == True:
            self.neuron_weights =  self.add_weight(shape=(self.number_of_factors, self.number_of_neurons), initializer='normal', trainable=True, name='neuron_weights', constraint=tf.keras.constraints.NonNeg())
            self.trial_weights =   self.add_weight(shape=(self.number_of_factors, self.trial_length),      initializer='normal', trainable=True, name='trial_weights', constraint=tf.keras.constraints.NonNeg())
            self.session_weights = self.add_weight(shape=(self.number_of_factors, self.number_of_trials),  initializer='normal', trainable=True, name='session_weights', constraint=tf.keras.constraints.NonNeg())
        else:
            # Create Weight Variables
            self.neuron_weights = self.add_weight(shape=(self.number_of_factors, self.number_of_neurons), initializer='normal', trainable=True, name='neuron_weights')
            self.trial_weights = self.add_weight(shape=(self.number_of_factors, self.trial_length), initializer='normal', trainable=True, name='trial_weights')
            self.session_weights = self.add_weight(shape=(self.number_of_factors, self.number_of_trials), initializer='normal', trainable=True, name='session_weights')

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
        for factor in range(self.number_of_factors):
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
    trial_length = np.shape(trial_tensors_list[0])[1]
    number_of_factors = 7
    model_list = []

    shared_trial_weights = tf.random.normal([number_of_factors, trial_length], 0, 1, tf.float32)

    # Create The Models
    for session in range(number_of_sessions):
        number_of_neurons = np.shape(trial_tensors_list[session])[0]
        number_of_timepoints = np.shape(trial_tensors_list[session])[1]
        number_of_trials = np.shape(trial_tensors_list[session])[2]

        model = tensor_decomposition_model(number_of_neurons, number_of_timepoints, number_of_trials, number_of_factors, non_negative=True)
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1))

        model_list.append(model)

    # Train The Models
    for epoch in range(100):
        print("Batch Epoch", epoch)
        for session_index in range(number_of_sessions):

            # Select Model
            model = model_list[session_index]

            # Set Weights To Be Shared Trial Weights
            model.trial_weights = shared_trial_weights

            # Fit Data
            training_data = [trial_tensors_list[session_index]]
            training_data = np.array(training_data)
            model.fit(training_data, epochs=100)

            # Save Weights
            shared_trial_weights = model.trial_weights

    return model_list, shared_trial_weights


def extract_tensors(file_list, trial_start, trial_stop):

    tensor_list = []
    blocks_list = []
    switches_list = []
    switch_classification_list = []
    session_name_list = []

    for matlab_file_location in file_list:

        # Get Session Name
        session_name = matlab_file_location.split('/')[-1]
        session_name = session_name.replace("_preprocessed_basic.mat", "")
        print("Extracting Tensor for Session: ", session_name)

        # Load Matalb Data
        data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

        # Extract Delta F Matrix
        delta_f_matrix = data_object.dF
        delta_f_matrix = np.nan_to_num(delta_f_matrix)
        delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
        delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)

        # Extract Switch Trials
        expected_odour_trials = data_object.mismatch_trials['exp_odour'][0]
        perfect_switch_trials = data_object.mismatch_trials['perfect_switch']


        visual_context_onsets = data_object.vis1_frames[0]
        odour_context_onsets = data_object.irrel_vis1_frames[0]
        all_onsets = np.concatenate([visual_context_onsets, odour_context_onsets])
        all_onsets.sort()

        # Get Trial Indexes Of Switches
        switch_indexes = []
        for trial in expected_odour_trials:
            index = list(all_onsets).index(trial)
            switch_indexes.append(index)

        # Get Block Boundaires
        visual_blocks, odour_blocks = get_block_boundaries(all_onsets, visual_context_onsets, odour_context_onsets)

        # Create Trial Tensor
        trial_tensor = create_trial_tensor(delta_f_matrix, all_onsets, trial_start, trial_stop)

        print("Trial Tensor Shape", np.shape(trial_tensor))

        # Add To Data to Lists
        tensor_list.append(trial_tensor)
        blocks_list.append([visual_blocks, odour_blocks])
        switches_list.append(switch_indexes)
        switch_classification_list.append(perfect_switch_trials)
        session_name_list.append(session_name)

    return tensor_list, blocks_list, switches_list, switch_classification_list, session_name_list




def plot_factors(trial_loadings, time_loadings, visual_blocks, odour_blocks, save_directory, session_name, trial_start, plot_switching=False, switch_indexes=None, perfect_switch_trials=None):

    number_of_factors = np.shape(trial_loadings)[0]
    rows = number_of_factors
    columns = 2

    print("Number of factors", number_of_factors)
    print("Time Loadings", np.shape(time_loadings))
    print("Trial Loadings", np.shape(trial_loadings))

    figure_count = 1
    figure_1 = plt.figure()
    figure_1.suptitle(session_name)
    for factor in range(number_of_factors):
        time_axis = figure_1.add_subplot(rows,  columns, figure_count)
        trial_axis = figure_1.add_subplot(rows, columns, figure_count + 1)
        figure_count += 2

        time_axis.set_title("Factor " + str(factor) + " Time Loadings")
        trial_axis.set_title("Factor " + str(factor) + " Trial Loadings")
        time_data = time_loadings[factor]
        trial_data = trial_loadings[factor]

        time_axis.plot(time_data)
        trial_axis.plot(trial_data, c='orange')

        # Plot Switch Trials
        if plot_switching == True:
            number_of_switches = len(switch_indexes)
            for switch_index in range(number_of_switches):
                switch_time = switch_indexes[switch_index]
                switch_type = perfect_switch_trials[switch_index]

                if switch_type == 1:
                    colour='m'
                else:
                    colour='k'

                trial_axis.vlines(switch_time, ymin=np.min(trial_data), ymax=np.max(trial_data), color=colour)


        # Mark Stimuli Onset
        time_axis.vlines(0-trial_start, ymin=np.min(time_data), ymax=np.max(time_data), color='k')

        # Highligh Blocks
        for block in visual_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='blue')
        for block in odour_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='green')

    figure_1.set_size_inches(18.5, 16)
    figure_1.tight_layout()
    plt.savefig(save_directory + "/" + session_name + ".png", dpi=200)
    plt.close()




os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.run_functions_eagerly(True)
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/"
file_list = load_matlab_sessions(base_directory)

# Extract Tensors
trial_start = -10
trial_stop = 40
tensor_list, blocks_list, switches_list, switch_classification_list, session_name_list = extract_tensors(file_list, trial_start=trial_start, trial_stop=trial_stop)

# Get Initial Alignment Matricies
#alignmnet_matricies = align_sessions(condition_tensor_list, number_of_factors)


# Train Model
model_list, shared_trial_weights = train_model(tensor_list)

# Plot Factors
save_directory = r"/home/matthew/Pictures/Shared_TCA_Plots/"
number_of_sessions = len(file_list)

for session in range(number_of_sessions):

    session_model = model_list[session]
    neuron_factors = session_model.neuron_weights
    session_factors = session_model.session_weights

    blocks = blocks_list[session]
    switch_indexes = switches_list[session]
    perfect_switch_list = switch_classification_list[session]
    session_name = session_name_list[session]

    plot_factors(session_factors, shared_trial_weights, blocks[0], blocks[1], save_directory, session_name, trial_start=trial_start, plot_switching=True, switch_indexes=switch_indexes, perfect_switch_trials=perfect_switch_list)



