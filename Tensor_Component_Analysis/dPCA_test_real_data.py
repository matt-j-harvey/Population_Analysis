import numpy as np
import matplotlib.pyplot as plt
from dPCA import dPCA
import os

import Load_Data_For_TCA


def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list


def get_trial_averages(tensor, condition_index_list):

    trial_averaged_tensor = []

    number_of_conditions = len(condition_index_list)
    for condition_index in range(number_of_conditions):
        condition_trial_indicies = condition_index_list[condition_index]

        condition_trial_list = []
        for trial_index in condition_trial_indicies:
            condition_trial_list.append(tensor[trial_index])

        condition_trial_list = np.array(condition_trial_list)
        condition_mean = np.mean(condition_trial_list, axis=0)
        condition_mean = np.subtract(condition_mean, np.mean(condition_mean))
        trial_averaged_tensor.append(condition_mean)

    trial_averaged_tensor = np.array(trial_averaged_tensor)
    return trial_averaged_tensor


def create_psudo_tensor(trial_tensor, indicies_list):

    """
    trialX - array - like, shape(n_trials, n_samples, n_features_1, n_features_2, ...)
    Trial - by - trial data.Shape is similar to X but with an additional axis at the beginning with different trials.
    If different combinations of features have different number of trials, then set n_samples to the maximum number of
    trials and fill unoccupied datapoints with NaN.
    """

    # Create Empty Nan Trial
    number_of_timepoints = np.shape(trial_tensor)[1]
    number_of_neurons = np.shape(trial_tensor)[2]
    nan_trial = np.empty((number_of_timepoints, number_of_neurons))
    nan_trial[:] = np.nan


    # Get Max Condition Size
    condition_size_list = []
    for condition in indicies_list:
        condition_size = len(condition)
        condition_size_list.append(condition_size)
    max_condition_size = np.max(condition_size_list)
    print("Max condition size", max_condition_size)

    pseudo_tensor = []
    number_of_conditions = len(indicies_list)
    for condition_index in range(number_of_conditions):
        condition_tensor = []
        condition_size = condition_size_list[condition_index]
        condition_indicies = indicies_list[condition_index]

        for trial_iterator in range(max_condition_size):
            if trial_iterator < condition_size:
                trial_index = condition_indicies[trial_iterator]
                trial_data = trial_tensor[trial_index]
            else:
                trial_data = nan_trial

            condition_tensor.append(trial_data)
        pseudo_tensor.append(condition_tensor)

    pseudo_tensor = np.array(pseudo_tensor)
    print("Pseudo ensor shape 1", np.shape(pseudo_tensor))

    # Samples Neurons Stimuli Time
    pseudo_tensor = np.moveaxis(pseudo_tensor, [0, 1, 2, 3], [2, 0, 3, 1])
    print("Pseudo Tensor Shape 2", np.shape(pseudo_tensor))
    return pseudo_tensor














# Set Number of Factors
number_of_factors = 10
trial_start = -6
trial_stop = 26 #Should be 18 if offset
onset_or_offset = 'onset'

# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/Best_switching_sessions_all_sites"
behaviour_matrix_directory = r"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Behaviour_Matricies"
file_list = load_matlab_sessions(base_directory)


for matlab_file_location in file_list:

    # Get Session Name
    session_name = matlab_file_location.split('/')[-1]
    session_name = session_name.replace("_preprocessed_basic.mat", "")
    print("Performing Tensor Component Analysis for Session: ", session_name)

    # Load Data
    trial_tensor, stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies = Load_Data_For_TCA.load_data_for_tca(matlab_file_location, behaviour_matrix_directory, trial_start, trial_stop, onset_or_offset)

    # Create Trial Average Tensor
    condition_index_list = [stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies + imperfect_transition_indicies]

    trial_averaged_tensor = get_trial_averages(trial_tensor, condition_index_list)
    print("Trial Average Tensor", np.shape(trial_averaged_tensor))

    pseudo_tensor = create_psudo_tensor(trial_tensor, condition_index_list)
    print("Pseudo Tensor", np.shape(pseudo_tensor))

    reshuffled_trial_averaged_tensor = np.moveaxis(trial_averaged_tensor, [0,1,2], [1,2,0])
    print("Reshuffled rial average tensor", np.shape(trial_averaged_tensor))

    # Neurons, Stimuli, Time
    dpca = dPCA.dPCA(labels='st', regularizer=None, n_components=10)
    dpca.protect = ['t']
    Z = dpca.fit_transform(reshuffled_trial_averaged_tensor, pseudo_tensor)

    # Plot Factors
    S = 3
    T = 32

    time = np.arange(T)
    for component in range(10):
        figure_1 = plt.figure(figsize=(16, 7))

        axis_1 = figure_1.add_subplot(131)
        axis_2 = figure_1.add_subplot(132)
        axis_3 = figure_1.add_subplot(133)


        for s in range(S):
            axis_1.plot(time, Z['t'][component, s])
            axis_1.set_title('1st time component')

        for s in range(S):
            axis_2.plot(time, Z['s'][component, s])
            axis_2.set_title('1st stimulus component')

        for s in range(S):
            axis_3.plot(time, Z['st'][component, s])
            axis_3.set_title('1st mixing component')

        plt.show()

