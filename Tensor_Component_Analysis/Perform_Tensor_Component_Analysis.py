import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import AffinityPropagation
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
from tensorly.decomposition import parafac, CP, non_negative_parafac
import os

import Import_Preprocessed_Data


def view_raster(delta_f_matrix):
    plt.imshow(delta_f_matrix)
    plt.show()


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

    return selected_data





def plot_factors(trial_loadings, time_loadings, switch_indexes, visual_blocks, odour_blocks, save_directory, session_name, trial_start, perfect_switch_trials):

    number_of_factors = np.shape(trial_loadings)[1]
    rows = number_of_factors
    columns = 2

    figure_count = 1
    figure_1 = plt.figure()
    for factor in range(number_of_factors):
        time_axis = figure_1.add_subplot(rows,  columns, figure_count)
        trial_axis = figure_1.add_subplot(rows, columns, figure_count + 1)
        figure_count += 2

        time_axis.set_title("Factor " + str(factor) + " Time Loadings")
        trial_axis.set_title("Factor " + str(factor) + " Trial Loadings")
        time_data = time_loadings[:, factor]
        trial_data = trial_loadings[:, factor]

        time_axis.plot(time_data)
        trial_axis.plot(trial_data, c='orange')

        # Plot Switch Trials
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




def smooth_delta_f_matrix(delta_f_maxtrix):

    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_delta_f = []

    for trace in delta_f_maxtrix:
        smoothed_trace = np.convolve(trace, kernel, mode='same')
        smoothed_delta_f.append(smoothed_trace)

    smoothed_delta_f = np.array(smoothed_delta_f)
    return smoothed_delta_f


def normalise_delta_f_matrix(delta_f_matrix):

    delta_f_matrix = np.transpose(delta_f_matrix)
    # Normalise Each Neuron to Min 0, Max 1

    #Subtract Min To Get Min = 0
    min_vector = np.min(delta_f_matrix, axis=0)
    delta_f_matrix = np.subtract(delta_f_matrix, min_vector)

    # Divide By Max To Get Max = 1
    max_vector = np.max(delta_f_matrix, axis=0)
    delta_f_matrix = np.divide(delta_f_matrix, max_vector)

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
    print("Nubmer of onsets", number_of_onsets)
    for onset_index in range(1, number_of_onsets):

        # Get Onset
        onset = combined_onsets[onset_index]
        print("Current Onset: ", onset)

        # If we are currently in an Visual Block
        if current_block_type == 0:
            print("CUrrent Block Visual")

            # If The Next Onset is An Odour Block - Block Finish, add Block To Boundaries
            if onset in odour_context_onsets:
                print("Weve got on odour onset")
                current_block_end = onset_index
                visual_blocks.append([current_block_start, current_block_end])
                current_block_type = 1
                current_block_start = onset_index

        # If we Are currently in an Odour BLock
        if current_block_type == 1:
            print("Current Block Olfactoerty")

            # If The NExt Onset Is a Visual Trial - BLock Finish Add Block To Block Boundaires
            if onset in visual_context_onsets:
                current_block_end = onset_index
                odour_blocks.append([current_block_start, current_block_end])
                current_block_type = 0
                current_block_start = onset_index

    print("Visua blocks", visual_blocks)
    print("Odour blocks", odour_blocks)
    return visual_blocks, odour_blocks








def perform_tensor_component_analysis(file_list, save_directory, trial_start=-10, trial_stop=42, number_of_factors=7):

    for matlab_file_location in file_list:

        # Get Session Name
        session_name = matlab_file_location.split('/')[-1]
        session_name = session_name.replace("_preprocessed_basic.mat", "")
        print("Performing Tensor Component Analysis for Session: ", session_name)

        # Load Matalb Data
        data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

        # Extract Delta F Matrix
        delta_f_matrix = data_object.dF
        delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
        delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)

        # Extract Visual Onsets
        visual_context_vis_1_onsets = data_object.vis1_frames[0]
        odour_context_vis_1_onsets  = data_object.irrel_vis1_frames[0]
        all_vis_1_onsets = np.concatenate([visual_context_vis_1_onsets, odour_context_vis_1_onsets])
        all_vis_1_onsets.sort()
        expected_odour_trials = data_object.mismatch_trials['exp_odour'][0]
        perfect_switch_trials = data_object.mismatch_trials['perfect_switch']


        # Get Trial Indexes Of Switches
        switch_indexes = []
        for trial in expected_odour_trials:
            index = list(all_vis_1_onsets).index(trial)
            switch_indexes.append(index)
        print("Switch Indexes", switch_indexes)

        # Get Block Boundaires
        print("Combined", all_vis_1_onsets)
        print("Visual ", visual_context_vis_1_onsets)
        print("Odour", odour_context_vis_1_onsets)
        visual_blocks, odour_blocks = get_block_boundaries(all_vis_1_onsets, visual_context_vis_1_onsets, odour_context_vis_1_onsets)


        # Create Trial Tensor
        vis_1_trial_tensor = create_trial_tensor(delta_f_matrix, all_vis_1_onsets, trial_start, trial_stop)
        print("Trial Tensor Shape: ", np.shape(vis_1_trial_tensor))

        # Perform Tensor Decomposition
        weights, factors = non_negative_parafac(vis_1_trial_tensor, rank=number_of_factors, init='svd', verbose=1, n_iter_max=250)
        print("Tensor shape", np.shape(factors))

        trial_loadings = factors[0]
        time_loadings = factors[1]
        neuron_loadings = factors[2]


        plot_factors(trial_loadings, time_loadings, switch_indexes, visual_blocks, odour_blocks, save_directory, session_name, trial_start, perfect_switch_trials)


# Load Matlab Data

# Load Matlab Data
file_list = ["/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201022_112044__ACV004_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201016_113151__ACV004_B2_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201026_103629__ACV014_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201024_104327__ACV005_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201026_122511__ACV011_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201103_160924__ACV013_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201029_145825__ACV011_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201021_121703__ACV005_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20201027_140620__ACV013_B3_SWITCH_preprocessed_basic.mat",
             "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Combined_Sessions/20200922_114059__ACV003_B3_SWITCH_preprocessed_basic.mat"]


save_directory = "/home/matthew/Pictures/TCA_Plots/"
perform_tensor_component_analysis(file_list, save_directory, number_of_factors=7)
