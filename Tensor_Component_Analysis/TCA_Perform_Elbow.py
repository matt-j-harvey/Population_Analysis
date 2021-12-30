import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import AffinityPropagation
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
from tensorly.decomposition import parafac, CP, non_negative_parafac
import tensorly as tl
import os

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



def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list




def get_tca_elbow(matlab_file_location, trial_start=-10, trial_stop=42, factor_start=1, factor_stop=20):

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

    # Get Delta_F_Shape
    number_of_neurons = np.shape(delta_f_matrix)[0]
    number_of_timepoints = np.shape(delta_f_matrix)[1]

    # Extract Visual Onsets
    visual_context_onsets = data_object.vis1_frames[0]
    odour_context_onsets  = data_object.irrel_vis1_frames[0]
    all_onsets = np.concatenate([visual_context_onsets, odour_context_onsets])
    all_onsets.sort()

    # Create Trial Tensor
    trial_tensor = create_trial_tensor(delta_f_matrix, all_onsets, trial_start, trial_stop)
    factor_list = []
    error_list = []

    for factor_number in range(factor_start, factor_stop):

        # Perform Tensor Decomposition
        weights, factors = non_negative_parafac(trial_tensor, rank=factor_number, init='svd', verbose=0, n_iter_max=250)

        # Reconstruct Original Tensor
        reconstruction = tl.cp_to_tensor((weights, factors))

        # Get Error
        error = np.subtract(trial_tensor, reconstruction)
        error = np.abs(error)
        error = np.sum(error)

        # Normalise Error
        error = np.divide(error, (number_of_neurons * number_of_timepoints))
        print("Factors: ", factor_number, "Error: ", error)

        # Add To List
        factor_list.append(factor_number)
        error_list.append(error)

    return factor_list, error_list


# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/"
save_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/Elbow_Method"
file_list = load_matlab_sessions(base_directory)

error_meta_list = []
factor_meta_list = []

"""
# Perform TCA Elbow
for file in file_list:

    # Get TCA Elbow
    factor_list, error_list = get_tca_elbow(file)

    # Save These Lists
    factor_meta_list.append(factor_list)
    error_meta_list.append(error_list)

np.save(os.path.join(save_directory, "Factor_Meta_List.npy"), factor_meta_list)
np.save(os.path.join(save_directory, "Error_Meta_List.npy"), error_meta_list)
"""
# Plot These
factor_meta_list = np.load(os.path.join(save_directory, "Factor_Meta_List.npy"))
error_meta_list = np.load(os.path.join(save_directory, "Error_Meta_List.npy"))

for file_index in range(len(factor_meta_list)):
    normalised_error_list = np.divide(error_meta_list[file_index], error_meta_list[file_index][0])
    plt.plot(factor_meta_list[file_index], normalised_error_list)
plt.show()

