import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import statsmodels.api as sm

import Import_Preprocessed_Data


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


def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list


def create_trial_regressors(matlab_file_location, number_of_trials):

    # Load Matalb Data
    data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

    # Create Holders For Regressors
    visual_context_regressor = np.zeros(number_of_trials)
    odour_context_regressor = np.zeros(number_of_trials)
    transition_regressor = np.zeros(number_of_trials)

    # Extract Trial Onsets
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

    # Create Regressors
    for block in visual_blocks:
        block_start = block[0]
        block_stop = block[1]
        visual_context_regressor[block_start:block_stop] = 1

    transition_count = 0
    for block in odour_blocks:
        block_start = block[0]
        block_stop = block[1]
        odour_context_regressor[block_start:block_stop] = 1
        transition_regressor[block_stop + 1: block_stop + 2] = 1
        transition_count += 1

    # Return Regressors
    return visual_context_regressor, odour_context_regressor, transition_regressor


# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/Best_switching_sessions_all_sites"
file_list = load_matlab_sessions(base_directory)
p_value_cutoff = 0.05

group_average_temporal_component_list = []

for matlab_file_location in file_list:

    # Get Session Name
    session_name = matlab_file_location.split('/')[-1]
    session_name = session_name.replace("_preprocessed_basic.mat", "")
    print("Performing Tensor Regression for Session: ", session_name)

    # Get Save Directory
    factor_save_directory = base_directory + "/" + session_name

    # Load Trial Factors
    trial_factors = np.load(os.path.join(factor_save_directory, "trial_loadings.npy"))
    time_factors = np.load(os.path.join(factor_save_directory, "time_loadings.npy"))

    number_of_trials = np.shape(trial_factors)[0]
    number_of_factors = np.shape(trial_factors)[1]

    print("Trial Factors", np.shape(trial_factors))
    print("Time Factor", np.shape(time_factors))


    # Create Stimuli Regressors
    visual_context_regressor, odour_context_regressor, transition_regressor = create_trial_regressors(matlab_file_location, number_of_trials)

    """
    plt.plot(visual_context_regressor, c='b')
    plt.plot(odour_context_regressor, c='g')
    plt.plot(transition_regressor, c='m')
    plt.show()
    """

    # Perform Regression
    selected_regressor = transition_regressor

    trial_factors = sm.add_constant(trial_factors)
    est = sm.OLS(selected_regressor, trial_factors)
    est2 = est.fit()
    print(est2.summary())

    # Get Significantly Modulated Factors
    significantly_modulated_factors = []
    for factor_index in range(number_of_factors):
        p_value = est2.pvalues[factor_index]
        if p_value < p_value_cutoff:
            significantly_modulated_factors.append(factor_index)

    # Get Average Temporal Component
    average_temporal_component_list = []
    average_trial_component_list = []
    parameters = est2.params
    print(len(parameters))
    for signficant_factor in significantly_modulated_factors:

        # Get Factor Coefficient
        factor_coefficient = parameters[signficant_factor]

        # Print Factor Info
        print("Factor: ", signficant_factor)
        print("Coef: ", factor_coefficient)

        # Extract Factor Loadings
        selected_trial_loadings = trial_factors[:, signficant_factor]
        selected_time_loadings = time_factors[:, signficant_factor]

        # Multiply Factor By Regression Coefficients
        selected_trial_loadings = np.multiply(selected_trial_loadings, factor_coefficient)
        selected_time_loadings = np.multiply(selected_time_loadings, factor_coefficient)

        average_trial_component_list.append(selected_trial_loadings)
        average_temporal_component_list.append(selected_time_loadings)

    average_temporal_component_list = np.array(average_temporal_component_list)
    average_trial_component_list = np.array(average_trial_component_list)

    average_temporal_component = np.mean(average_temporal_component_list, axis=0)
    average_trial_component = np.mean(average_trial_component_list, axis=0)


    plt.plot(average_temporal_component)
    plt.show()

    plt.plot(np.multiply(selected_regressor, np.max(average_trial_component)), alpha=0.5)
    plt.plot(average_trial_component, alpha=0.5)
    plt.show()

    normalised_component = np.subtract(average_temporal_component, np.min(average_temporal_component))
    normalised_component = np.divide(normalised_component, np.max(normalised_component))
    group_average_temporal_component_list.append(normalised_component)

for component in group_average_temporal_component_list:
    plt.plot(component)
plt.show()

mean_temporal_component = np.mean(group_average_temporal_component_list, axis=0)
temporal_component_sd = np.std(group_average_temporal_component_list, axis=0)
print(len(temporal_component_sd))

x_values = list(range(len(mean_temporal_component)))
plt.plot(mean_temporal_component)
plt.fill_between(x=x_values, y1=mean_temporal_component, y2=np.add(mean_temporal_component, temporal_component_sd),         alpha=0.5)
plt.fill_between(x=x_values, y1=mean_temporal_component, y2=np.subtract(mean_temporal_component, temporal_component_sd),    alpha=0.5)
plt.show()



