import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import AffinityPropagation
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
from tensorly.decomposition import parafac, CP, non_negative_parafac
import os

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Import_Preprocessed_Data
import Widefield_General_Functions




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



def get_transitions_from_behaviour_matrix(session_name, behaviour_matrix_directory):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(behaviour_matrix_directory, session_name + "_preprocessed_basic", "Behaviour_Matrix.npy"), allow_pickle=True)
    print("Behaviour matrix", np.shape(behaviour_matrix))


    # Get Selected Trials
    stable_odour_vis_1 = []
    perfect_transition_vis_1 = []
    imperfect_transition_vis_1 = []
    stable_visual_vis_1 = []


    number_of_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(1, number_of_trials):

        # Load Trial Data
        trial_data = behaviour_matrix[trial_index]

        # Is Trial In Visual Block
        trial_type = trial_data[1]
        if trial_type == 1 or trial_type == 2:

            # Is Trial First In Block
            first_in_block = trial_data[9]
            if first_in_block == 1:

                # Is Trial A Miss
                lick_response = trial_data[2]
                if lick_response == 0:

                    # Check If Perfect Transition (Following Trial Is Correct)
                    if behaviour_matrix[trial_index + 1][3] == 1:
                        perfect_transition_vis_1.append(trial_index)

                        # Add Following Trial As Stable Vis 1
                        stable_visual_vis_1.append(trial_index + 1)

                    else:
                        imperfect_transition_vis_1.append(trial_index)

                    # Add Trial Before To Stable Odour
                    stable_odour_vis_1.append(trial_index - 1)

    # Get Selected Onsets
    stable_odour_vis_1_onsets = []
    perfect_transition_vis_1_onsets = []
    imperfect_transition_vis_1_onsets = []
    stable_visual_vis_1_onsets = []

    for trial in stable_odour_vis_1:
        stable_odour_vis_1_onsets.append(behaviour_matrix[trial][13])

    for trial in perfect_transition_vis_1:
        perfect_transition_vis_1_onsets.append(behaviour_matrix[trial][11])

    for trial in imperfect_transition_vis_1:
        imperfect_transition_vis_1_onsets.append(behaviour_matrix[trial][11])

    for trial in stable_visual_vis_1:
        stable_visual_vis_1_onsets.append(behaviour_matrix[trial][11])


    # Combine Into All Onsets
    all_onsets = stable_odour_vis_1_onsets + perfect_transition_vis_1_onsets + imperfect_transition_vis_1_onsets + stable_visual_vis_1_onsets
    all_onsets.sort()




    return all_onsets, stable_odour_vis_1_onsets, stable_visual_vis_1_onsets, perfect_transition_vis_1_onsets, imperfect_transition_vis_1_onsets


def get_trial_type_labels(all_onsets, stable_odour_vis_1_onsets, stable_visual_vis_1_onsets, perfect_transition_vis_1_onsets, imperfect_transition_vis_1_onsets):

    # Create Label List
    stable_odour_1_indexes = []
    perfect_transition_indexes = []
    imperfect_transition_indexes = []
    stable_visual_1_indexes = []

    number_of_onsets = len(all_onsets)
    for onset_index in range(number_of_onsets):
        onset = all_onsets[onset_index]

        if onset in stable_odour_vis_1_onsets:
            stable_odour_1_indexes.append(onset_index)

        if onset in perfect_transition_vis_1_onsets:
            perfect_transition_indexes.append(onset_index)

        if onset in imperfect_transition_vis_1_onsets:
            imperfect_transition_indexes.append(onset_index)

        if onset in stable_visual_vis_1_onsets:
            stable_visual_1_indexes.append(onset_index)

    return stable_odour_1_indexes, perfect_transition_indexes, imperfect_transition_indexes, stable_visual_1_indexes


def plot_factors(trial_loadings, time_loadings, trial_start, stable_odour_1_indexes, perfect_transition_indexes, imperfect_transition_indexes, stable_visual_1_indexes):

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

        # Mark Stimuli Onset
        time_axis.vlines(0-trial_start, ymin=np.min(time_data), ymax=np.max(time_data), color='k')

        # Scatter Trial Types
        trial_axis.scatter(stable_odour_1_indexes,         np.multiply(np.ones(len(stable_odour_1_indexes)),       np.max(trial_loadings[:, factor])), c='g')
        trial_axis.scatter(perfect_transition_indexes,      np.multiply(np.ones(len(perfect_transition_indexes)),   np.max(trial_loadings[:, factor])), c='m')
        trial_axis.scatter(imperfect_transition_indexes,    np.multiply(np.ones(len(imperfect_transition_indexes)), np.max(trial_loadings[:, factor])), c='r')
        trial_axis.scatter(stable_visual_1_indexes,         np.multiply(np.ones(len(stable_visual_1_indexes)),      np.max(trial_loadings[:, factor])), c='b')


    figure_1.set_size_inches(18.5, 16)
    figure_1.tight_layout()
    plt.show()






def load_transition_triplets(file_list, behaviour_matrix_directory):

    for matlab_file_location in file_list:

        # Get Session Name
        session_name = matlab_file_location.split('/')[-1]
        session_name = session_name.replace("_preprocessed_basic.mat", "")
        print("Performing Tensor Component Analysis for Session: ", session_name)

        # Load Matalb Data
        data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

        # Get Onsets From Beahviour Matrix
        all_onsets, stable_odour_vis_1_onsets, stable_visual_vis_1_onsets, perfect_transition_vis_1_onsets, imperfect_transition_vis_1_onsets = get_transitions_from_behaviour_matrix(session_name, behaviour_matrix_directory)

        # Get Trial Type Indexes
        stable_odour_1_indexes, perfect_transition_indexes, imperfect_transition_indexes, stable_visual_1_indexes = get_trial_type_labels(all_onsets, stable_odour_vis_1_onsets, stable_visual_vis_1_onsets, perfect_transition_vis_1_onsets, imperfect_transition_vis_1_onsets)

        # Extract Delta F Matrix
        delta_f_matrix = data_object.dF
        delta_f_matrix = np.nan_to_num(delta_f_matrix)
        delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
        delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)

        # Get Trial Tensor
        trial_start = -6
        trial_stop = 24
        trial_tensor = create_trial_tensor(delta_f_matrix, all_onsets, trial_start, trial_stop)

        # Perform Tensor Decomposition
        number_of_factors = 5
        weights, factors = non_negative_parafac(trial_tensor, rank=number_of_factors, init='svd', verbose=1, n_iter_max=250)
        #weights, factors = parafac(trial_tensor, rank=number_of_factors, init='svd', verbose=1, n_iter_max=1000)

        trial_loadings = factors[0]
        time_loadings = factors[1]
        neuron_loadings = factors[2]

        # Plot Factors
        plot_factors(trial_loadings, time_loadings, trial_start,  stable_odour_1_indexes, perfect_transition_indexes, imperfect_transition_indexes, stable_visual_1_indexes)




# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/Best_switching_sessions_all_sites"
behaviour_matrix_directory = r"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Behaviour_Matricies"
file_list = load_matlab_sessions(base_directory)
load_transition_triplets(file_list, behaviour_matrix_directory)
