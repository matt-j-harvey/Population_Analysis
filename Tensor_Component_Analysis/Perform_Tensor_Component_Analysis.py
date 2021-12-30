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




def plot_factors_combined(trial_loadings, time_loadings, visual_blocks, odour_blocks, save_directory, session_name, trial_start, vis_1_indexes, vis_2_indexes, plot_switching=False, switch_indexes=None, perfect_switch_trials=None):

    number_of_factors = np.shape(trial_loadings)[1]
    rows = number_of_factors
    columns = 2

    figure_count = 1
    figure_1 = plt.figure()
    figure_1.suptitle(session_name)
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

        # Scatter Trials
        print("Trial Data Shape", np.shape(trial_data))
        vis_1_points = []
        for index in vis_1_indexes:
            vis_1_points.append(trial_data[index])

        vis_2_points = []
        for index in vis_2_indexes:
            vis_2_points.append(trial_data[index])

        trial_axis.scatter(vis_1_indexes, vis_1_points, c='b', alpha=0.5)
        trial_axis.scatter(vis_2_indexes, vis_2_points, c='r', alpha=0.5)


    figure_1.set_size_inches(18.5, 16)
    figure_1.tight_layout()
    plt.savefig(save_directory + "/" + session_name + ".png", dpi=200)
    plt.close()







def plot_factors(trial_loadings, time_loadings, visual_blocks, odour_blocks, save_directory, session_name, trial_start, plot_switching=False, switch_indexes=None, perfect_switch_trials=None):

    number_of_factors = np.shape(trial_loadings)[1]
    rows = number_of_factors
    columns = 2

    figure_count = 1
    figure_1 = plt.figure()
    figure_1.suptitle(session_name)
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








def perform_tensor_component_analysis(file_list, base_directory, plot_save_directory, stimuli="vis_1", trial_start=-10, trial_stop=48, number_of_factors=7, offset=1):

    for matlab_file_location in file_list:

        # Get Session Name
        session_name = matlab_file_location.split('/')[-1]
        session_name = session_name.replace("_preprocessed_basic.mat", "")
        print("Performing Tensor Component Analysis for Session: ", session_name)

        # Load Matalb Data
        data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

        # Extract Delta F Matrix
        delta_f_matrix = data_object.dF
        delta_f_matrix = np.nan_to_num(delta_f_matrix)
        delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
        delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)

        # Extract Switch Trials
        expected_odour_trials = data_object.mismatch_trials['exp_odour'][offset]
        perfect_switch_trials = data_object.mismatch_trials['perfect_switch']

        # Extract Visual Onsets
        if stimuli=='vis_1':
            visual_context_onsets = data_object.vis1_frames[offset]
            odour_context_onsets  = data_object.irrel_vis1_frames[offset]
            all_onsets = np.concatenate([visual_context_onsets, odour_context_onsets])
            all_onsets.sort()

            # Get Trial Indexes Of Switches
            switch_indexes = []
            for trial in expected_odour_trials:
                index = list(all_onsets).index(trial)
                switch_indexes.append(index)

        if stimuli=='vis_2':
            visual_context_onsets = data_object.vis2_frames[offset]
            odour_context_onsets  = data_object.irrel_vis2_frames[offset]
            all_onsets = np.concatenate([visual_context_onsets, odour_context_onsets])
            all_onsets.sort()

        if stimuli == 'all':
            visual_context_onsets_vis_1 = data_object.vis1_frames[offset]
            visual_context_onsets_vis_2 = data_object.vis2_frames[offset]
            odour_context_onsets_vis_1  = data_object.irrel_vis1_frames[offset]
            odour_context_onsets_vis_2  = data_object.irrel_vis2_frames[offset]

            all_onsets = np.concatenate([visual_context_onsets_vis_1, visual_context_onsets_vis_2, odour_context_onsets_vis_1, odour_context_onsets_vis_2])
            all_onsets.sort()

            visual_context_onsets = np.concatenate([visual_context_onsets_vis_1, visual_context_onsets_vis_2])
            visual_context_onsets.sort()

            odour_context_onsets = np.concatenate([odour_context_onsets_vis_1, odour_context_onsets_vis_2])
            odour_context_onsets.sort()

            all_vis_1_onsets = np.concatenate([visual_context_onsets_vis_1, odour_context_onsets_vis_1])
            all_vis_2_onsets = np.concatenate([visual_context_onsets_vis_2, odour_context_onsets_vis_2])

            # Get Trial Indexes Of Switches
            switch_indexes = []
            for trial in expected_odour_trials:
                index = list(all_onsets).index(trial)
                switch_indexes.append(index)

            # Get Trial Indexes Of Vis 1
            vis_1_indexes = []
            for trial in all_vis_1_onsets:
                if trial + trial_stop < np.shape(delta_f_matrix)[1]:
                    index = list(all_onsets).index(trial)
                    vis_1_indexes.append(index)

            # Get Trial Indexes of Vis 2
            vis_2_indexes = []
            for trial in all_vis_2_onsets:
                if trial + trial_stop < np.shape(delta_f_matrix)[1]:
                    index = list(all_onsets).index(trial)
                    vis_2_indexes.append(index)

        print("Nubmer of onsets", len(list(all_onsets)))

        # Get Block Boundaires
        visual_blocks, odour_blocks = get_block_boundaries(all_onsets, visual_context_onsets, odour_context_onsets)

        # Create Trial Tensor
        trial_tensor = create_trial_tensor(delta_f_matrix, all_onsets, trial_start, trial_stop)
        print("Trial Tensor Shape", np.shape(trial_tensor))

        # Perform Tensor Decomposition
        weights, factors = non_negative_parafac(trial_tensor, rank=number_of_factors, init='svd', verbose=1, n_iter_max=250)
        #weights, factors = parafac(trial_tensor, rank=number_of_factors, init='svd', verbose=1, n_iter_max=250)

        # Save Factors
        factor_save_directory = base_directory + "/" + session_name
        if not os.path.exists(factor_save_directory):
            os.mkdir(factor_save_directory)

        trial_loadings = factors[0]
        time_loadings = factors[1]
        neuron_loadings = factors[2]

        np.save(os.path.join(factor_save_directory, "trial_loadings.npy"),  trial_loadings)
        np.save(os.path.join(factor_save_directory, "time_loadings.npy"),   time_loadings)
        np.save(os.path.join(factor_save_directory, "neuron_loadings.npy"), neuron_loadings)
        np.save(os.path.join(factor_save_directory, "all_onsets.npy"),  all_onsets)
        np.save(os.path.join(factor_save_directory, "perfect_switch_trials.npy"),  perfect_switch_trials)
        np.save(os.path.join(factor_save_directory, "visual_blocks.npy"),  visual_blocks)
        np.save(os.path.join(factor_save_directory, "odour_blocks.npy"),  odour_blocks)
        np.save(os.path.join(factor_save_directory, "switch_indicies.npy"), switch_indexes)

        if stimuli == "vis_1":
            plot_factors(trial_loadings, time_loadings, visual_blocks, odour_blocks, plot_save_directory, session_name, trial_start, plot_switching=True, switch_indexes=switch_indexes, perfect_switch_trials=perfect_switch_trials)
        elif stimuli == "vis_2":
            plot_factors(trial_loadings, time_loadings, visual_blocks, odour_blocks, plot_save_directory, session_name, trial_start)
        elif stimuli == "all":
            plot_factors_combined(trial_loadings, time_loadings, visual_blocks, odour_blocks, plot_save_directory, session_name, trial_start, vis_1_indexes, vis_2_indexes, plot_switching=True, switch_indexes=switch_indexes, perfect_switch_trials=perfect_switch_trials)


def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list


# Set Number of Factors
number_of_factors = 7

# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/"
file_list = load_matlab_sessions(base_directory)



# Perform TCA On All Vis Onsets
plot_save_directory = "/home/matthew/Pictures/TCA_Plots/Combined_TCA/"
perform_tensor_component_analysis(file_list, base_directory, plot_save_directory, stimuli='all', number_of_factors=7)

# Perform TCA On Vis 1 Onsets
#plot_save_directory = "/home/matthew/Pictures/TCA_Plots/Vis_1_TCA/"
#perform_tensor_component_analysis(file_list, base_directory, plot_save_directory, number_of_factors=number_of_factors)

# Perform TCA On VIs 2 Onsets
#plot_save_directory = "/home/matthew/Pictures/TCA_Plots/Vis_2_TCA/"
#perform_tensor_component_analysis(file_list, base_directory, plot_save_directory, stimuli='vis_2', number_of_factors=7)
