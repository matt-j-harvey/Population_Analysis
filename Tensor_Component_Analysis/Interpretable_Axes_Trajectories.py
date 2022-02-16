import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import AffinityPropagation
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
from tensorly.decomposition import parafac, CP, non_negative_parafac
import os
import matplotlib.gridspec as gridspec
import seaborn as sns
import Import_Preprocessed_Data



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



def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list




def create_trial_tensor(delta_f_matrix, onsets, start_window, stop_window):

    # Given A List Of Trial Onsets - Create A 3 Dimensional Tensor (Trial x Neuron x Trial_Aligned_Timepoint)
    number_of_timepoints = np.shape(delta_f_matrix)[1]

    # Transpose Delta F Matrix So Its Time x Neurons
    delta_f_matrix = np.transpose(delta_f_matrix)

    selected_data = []
    for onset in onsets:
        onset = int(onset)
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_stop < number_of_timepoints:
            trial_data = delta_f_matrix[int(trial_start):int(trial_stop)]
            selected_data.append(trial_data)

    selected_data = np.array(selected_data)

    return selected_data



def perform_interpretable_axis(file_list, trial_start=0, trial_stop=50):

    save_directory = "/home/matthew/Pictures/interpretable_axes_plots/"


    for matlab_file_location in file_list:

        # Get Session Name
        session_name = matlab_file_location.split('/')[-1]
        session_name = session_name.replace("_preprocessed_basic.mat", "")
        print("Getting Interpretable Axes: ", session_name)

        # Load Matalb Data
        data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

        # Extract Delta F Matrix
        delta_f_matrix = data_object.dF
        delta_f_matrix = np.nan_to_num(delta_f_matrix)
        delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
        delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)

        # Extract Trials
        visual_context_vis_1_onsets = data_object.vis1_frames[1]
        visual_context_vis_2_onsets = data_object.vis2_frames[1]
        odour_context_vis_1_onsets = data_object.irrel_vis1_frames[1]
        odour_context_vis_2_onsets = data_object.irrel_vis2_frames[1]

        # Package Onsets
        all_visual_context_onsets = np.concatenate([visual_context_vis_1_onsets, visual_context_vis_2_onsets])
        all_odour_context_onsets = np.concatenate([odour_context_vis_1_onsets, odour_context_vis_2_onsets])

        # Create Context Vector
        visual_trial_tensor = create_trial_tensor(delta_f_matrix, all_visual_context_onsets, trial_start, trial_stop)
        odour_trial_tensor = create_trial_tensor(delta_f_matrix, all_odour_context_onsets, trial_start, trial_stop)

        # Take Mean Across Trials
        mean_visual_trial_response = np.mean(visual_trial_tensor, axis=0)
        mean_odour_trial_response = np.mean(odour_trial_tensor, axis=0)

        # Take Mean Across Timepoints
        mean_visual_trial_response = np.mean(mean_visual_trial_response, axis=0)
        mean_odour_trial_response = np.mean(mean_odour_trial_response, axis=0)

        # Normalise These Vectors
        olfatory_vector_norm = np.linalg.norm(mean_odour_trial_response)
        olfatory_vector = np.divide(mean_odour_trial_response, olfatory_vector_norm)

        visual_vector_norm = np.linalg.norm(mean_visual_trial_response)
        visual_vector = np.divide(mean_visual_trial_response, visual_vector_norm)


        # Extract Switch Trials
        expected_odour_trials = data_object.mismatch_trials['exp_odour'][1]
        perfect_switch_trials = data_object.mismatch_trials['perfect_switch']

        number_of_switches = len(expected_odour_trials)

        for switch in range(number_of_switches):
            switch_onset = int(expected_odour_trials[switch])


            switch_coords = []

            for timepoint in range(trial_start, trial_stop):

                switch_activity = delta_f_matrix[:, switch_onset+timepoint]
                olfactory_projection = np.dot(switch_activity, olfatory_vector)
                visual_projection = np.dot(switch_activity, visual_vector)

                coords = [olfactory_projection, visual_projection]
                switch_coords.append(coords)

            switch_type = perfect_switch_trials[switch]
            switch_coords = np.array(switch_coords)

            if switch_type == 1:
                plt.plot(switch_coords[:, 0], switch_coords[:, 1], c='r', alpha=0.4)
            elif switch_type == 0:
                plt.plot(switch_coords[:, 0], switch_coords[:, 1], c='b', alpha=0.4)

    plt.show()
    plt.savefig(save_directory + str(timepoint).zfill(3) + ".png")
    plt.close()








# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/"
file_list = load_matlab_sessions(base_directory)

perform_interpretable_axis(file_list)