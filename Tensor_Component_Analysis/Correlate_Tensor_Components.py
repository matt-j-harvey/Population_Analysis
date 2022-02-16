import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
from scipy import stats

import Import_Preprocessed_Data


def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list



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


def get_best_matching_factor(factor_matrix, regressor):

    correlation_list = []

    number_of_factors = np.shape(factor_matrix)[1]
    print("Factor Matrix", np.shape(factor_matrix))
    print("Regressor", np.shape(regressor))

    for factor_index in range(number_of_factors):
        correlation, p_value = stats.pearsonr(factor_matrix[:, factor_index], regressor)
        correlation_list.append(correlation)

    max_correlation = np.max(correlation_list)
    best_factor = correlation_list.index(max_correlation)

    return factor_matrix[:, best_factor]



# Aligned To Onset of Imaginary odour - shifted version equivalent to offset of vis 1



# Three Session Regressions
# Visual Context Step Function
# Odour Context Step Function
# Transition Function


# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/Best_switching_sessions_all_sites"
file_list = load_matlab_sessions(base_directory)

transition_regressor_list = []
transition_factor_list = []
transition_perfection_list = []




for matlab_file_location in file_list:

    # Get Session Name
    session_name = matlab_file_location.split('/')[-1]
    session_name = session_name.replace("_preprocessed_basic.mat", "")
    print("Performing Tensor Component Analysis for Session: ", session_name)

    # Load Matlab Data
    data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

    # Extract Switch Trials
    offset = 1
    expected_odour_trials = data_object.mismatch_trials['exp_odour_vis'][offset]
    perfect_switch_trials = data_object.mismatch_trials['perfect_switch']

    visual_context_onsets = data_object.vis1_frames[offset]
    odour_context_onsets = data_object.irrel_vis1_frames[offset]
    all_onsets = np.concatenate([visual_context_onsets, odour_context_onsets])
    all_onsets.sort()

    # Get Trial Indexes Of Switches
    switch_indexes = []
    for trial in expected_odour_trials:
        index = list(all_onsets).index(trial)
        switch_indexes.append(index)

    # Get Block Boundaires
    visual_blocks, odour_blocks = get_block_boundaries(all_onsets, visual_context_onsets, odour_context_onsets)

    print("visual blocks", visual_blocks)
    print("odour blocks", odour_blocks)
    print("Perfect switch trials", perfect_switch_trials)

    # Load Factors
    factor_save_directory = base_directory + "/" + session_name
    trial_loadings = np.load(os.path.join(factor_save_directory, "trial_loadings.npy"))

    number_of_trials = np.shape(trial_loadings)[0]
    number_of_factors = np.shape(trial_loadings)[1]


    # Create Regressors

    visual_context_regressor = np.zeros(number_of_trials)
    odour_context_regressor = np.zeros(number_of_trials)
    transition_regressor = np.zeros(number_of_trials)
    perfection_regressor = np.zeros(number_of_trials)


    for block in visual_blocks:
        block_start = block[0]
        block_stop = block[1]
        visual_context_regressor[block_start:block_stop] = 1

    transition_count = 0
    for block in odour_blocks:
        block_start = block[0]
        block_stop = block[1]
        odour_context_regressor[block_start:block_stop] = 1
        transition_regressor[block_stop + 1: block_stop + 3] = 1

        if transition_count < len(perfect_switch_trials):
            if perfect_switch_trials[transition_count] == 1:
                perfection_regressor[block_stop + 1: block_stop + 3] = 1

        transition_count += 1

    print("Number of blocks from odour list", len(odour_blocks))
    print("Number of blocks from perfect list", len(perfect_switch_trials))


    visual_context_factor = get_best_matching_factor(trial_loadings, visual_context_regressor)
    odour_context_factor = get_best_matching_factor(trial_loadings, odour_context_regressor)
    transition_factor = get_best_matching_factor(trial_loadings, transition_regressor)

    transition_regressor_list.append(transition_regressor)
    transition_factor_list.append(transition_factor)
    transition_perfection_list.append(perfection_regressor)

    """
    figure_1 = plt.figure()
    grid_spec = GridSpec(number_of_factors, 3, figure=figure_1)

    for factor_index in range(number_of_factors):
        axis = figure_1.add_subplot(grid_spec[factor_index, 0])

        factor = trial_loadings[:, factor_index]
        factor_max = np.max(factor)


        axis.plot(factor, c='k')
        axis.plot(np.multiply(visual_context_regressor, factor_max), c='b')
        axis.plot(np.multiply(odour_context_regressor, factor_max), c='g')
        axis.plot(np.multiply(transition_regressor, factor_max), c='m')


    visual_regressor_axis       = figure_1.add_subplot(grid_spec[0, 1])
    odour_regressor_axis        = figure_1.add_subplot(grid_spec[1, 1])
    transition_regressor_axis   = figure_1.add_subplot(grid_spec[2, 1])

    visual_factor_axis          = figure_1.add_subplot(grid_spec[0, 2])
    odour_factor_axis           = figure_1.add_subplot(grid_spec[1, 2])
    transition_factor_axis      = figure_1.add_subplot(grid_spec[2, 2])

    visual_regressor_axis.plot(visual_context_regressor)
    odour_regressor_axis.plot(odour_context_regressor)
    transition_regressor_axis.plot(transition_regressor)

    visual_factor_axis.plot(visual_context_factor)
    odour_factor_axis.plot(odour_context_factor)
    transition_factor_axis.plot(transition_factor)

    plt.show()
    """

number_of_sessions = len(file_list)


figure_1 = plt.figure()
rows = 7
columns = 2

for session_index in range(number_of_sessions):

    session_transition_regressor = transition_regressor_list[session_index]
    session_transition_factor = transition_factor_list[session_index]
    session_perfection_regressor = transition_perfection_list[session_index]

    session_transition_regressor = np.divide(session_transition_regressor, np.max(session_transition_regressor))
    session_transition_regressor = np.multiply(session_transition_regressor, np.max(session_transition_factor))

    session_perfection_regressor = np.divide(session_perfection_regressor, np.max(session_perfection_regressor))
    session_perfection_regressor = np.multiply(session_perfection_regressor, np.max(session_transition_factor))


    axis = figure_1.add_subplot(rows, columns, session_index + 1)
    axis.plot(session_transition_regressor, c='k', alpha=0.4)
    axis.plot(session_transition_factor, c='b')
    axis.plot(session_perfection_regressor, c='m', alpha=0.4)

plt.show()
