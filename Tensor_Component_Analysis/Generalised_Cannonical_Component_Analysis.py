import numpy as np
import matplotlib.pyplot as plt
import os

from mvlearn.datasets import load_UCImultifeature
from mvlearn.embed import GCCA


import Import_Preprocessed_Data
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


def perform_generalised_cannonical_component_analysis(file_list, behaviour_matrix_directory, number_of_factors, trial_start, trial_stop, onset_or_offset, save_directory):

    # Get Details
    trial_length = trial_stop - trial_start
    number_of_sessions = len(file_list)

    # Load Data For Each Session
    group_tensor_list = []
    full_tensor_list = []
    trial_type_index_list = []
    session_name_list = []
    all_vis_1_onsets_list = []
    perfect_transition_onsets_list = []
    delta_f_matrix_list = []
    imperfect_transition_onsets_list = []

    for matlab_file_location in file_list:

        # Get Session Name
        session_name = matlab_file_location.split('/')[-1]
        session_name = session_name.replace("_preprocessed_basic.mat", "")
        session_name_list.append(session_name)

        # Load Data
        trial_tensor, stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies, all_vis_1_onsets, perfect_transition_onsets, delta_f_matrix, imperfect_transition_onsets = Load_Data_For_TCA.load_data_for_tca(matlab_file_location, behaviour_matrix_directory, trial_start, trial_stop, onset_or_offset)
        print("Trial Tensor Shape", np.shape(trial_tensor))
        full_tensor_list.append(trial_tensor)
        trial_type_index_list.append([stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies])

        # Create Trial Average Tensor
        condition_index_list = [stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies + imperfect_transition_indicies]
        trial_averaged_tensor = get_trial_averages(trial_tensor, condition_index_list)
        print("Trial Average Tensor", np.shape(trial_averaged_tensor))

        # Flatten trial Average Tensor
        number_of_conditions = np.shape(trial_averaged_tensor)[0]
        trial_length = np.shape(trial_averaged_tensor)[1]
        number_of_neurons = np.shape(trial_averaged_tensor)[2]
        flattened_trial_average_tensor = np.ndarray.reshape(trial_averaged_tensor, (number_of_conditions * trial_length, number_of_neurons))
        print("Flattend Trial Average Tensor", np.shape(trial_averaged_tensor))

        # Add To Group Tensor
        group_tensor_list.append(flattened_trial_average_tensor)

        # Add Onsets and Raw Matrix To List
        all_vis_1_onsets_list.append(all_vis_1_onsets)
        perfect_transition_onsets_list.append(perfect_transition_onsets)
        delta_f_matrix_list.append(delta_f_matrix)
        imperfect_transition_onsets_list.append(imperfect_transition_onsets)


        #  Visluase Ai Data
        ######visualise_ai_data(matlab_file_location, perfect_transition_onsets)

    gcca = GCCA(n_components=10)
    #gcca = GCCA(fraction_var=0.99)

    # Fit Model and Transform Data
    transformed_data = gcca.fit_transform(group_tensor_list)

    # Get Projection Matricies
    projection_matrixies = gcca.projection_mats_

    # Save Output
    gcca_output_dictionary = {
        "Model":gcca,
        "Transformed_Trial_Averages": transformed_data,
        "Projection_Matricies":projection_matrixies,
        "Session_Names":session_name_list,
        "Trial_Type_Indexes":trial_type_index_list,
        "Full_Tensor_List":full_tensor_list,
        "Trial_Start": trial_start,
        "Trial_Stop": trial_stop,
        "Delta_F_Matrix": delta_f_matrix_list,
        "All_Vis_1_Onsets": all_vis_1_onsets_list,
        "Perfect_Transition_Onsets":perfect_transition_onsets_list,
        "Imperfect_Transition_Onsets":imperfect_transition_onsets_list,
    }

    np.save(os.path.join(save_directory, "GCCA_Output_Dictionary.npy"), gcca_output_dictionary)


def load_ai_recorder_file(matlab_file_location):

    # Get Session Name
    session_name = matlab_file_location.split('/')[-1]
    session_name = session_name.replace("_preprocessed_basic.mat", "")
    print("Creating Behaviour Matrix for Session: ", session_name)

    # Load Matalb Data
    data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

    # Load Downsampled AI
    data_matrix = data_object.downsampled_AI['data']

    return data_matrix


def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Reward": 1,
        "Lick": 2,
        "Visual 1": 3,
        "Visual 2": 4,
        "Odour 1": 5,
        "Odour 2": 6,
        "Irrelevance": 7,
        "Running": 8,
        "Trial End": 9,
    }
    return channel_index_dictionary



def visualise_ai_data(matlab_file_location, perfect_transition_onsets):



    # Load Matalb Data
    ai_data_matrix = load_ai_recorder_file(matlab_file_location)

    # Create Stimuli Dictionary
    stimuli_dictionary = create_stimuli_dictionary()

    pre_window = 100
    post_window = 100

    for onset in perfect_transition_onsets:
        vis_1_trace = ai_data_matrix[stimuli_dictionary["Visual 1"]][onset-pre_window:onset+post_window]
        lick_trace = ai_data_matrix[stimuli_dictionary["Lick"]][onset-pre_window:onset+post_window]
        odour_1_trace = ai_data_matrix[stimuli_dictionary["Odour 1"]][onset-pre_window:onset+post_window]
        odour_2_trace = ai_data_matrix[stimuli_dictionary["Odour 2"]][onset-pre_window:onset+post_window]

        plt.vlines(pre_window, ymin=0, ymax=4, color='k')

        plt.plot(vis_1_trace)
        plt.plot(lick_trace)
        plt.plot(odour_1_trace, c='g')
        plt.plot(odour_2_trace, c='g')
        plt.show()

    print(np.shape(ai_data_matrix))



# Set Number of Factors
number_of_factors = 10
trial_start = -6
trial_stop = 17 #prev 17
onset_or_offset = 'offset'

# Load Matlab Data
base_directory = "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Best_switching_sessions_all_sites"
behaviour_matrix_directory = r"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Behaviour_Matricies"
save_directory = "/home/matthew/Documents/GCCA_Analysis"
file_list = load_matlab_sessions(base_directory)

# Perform GCCA
perform_generalised_cannonical_component_analysis(file_list, behaviour_matrix_directory, number_of_factors, trial_start, trial_stop, onset_or_offset, save_directory)

