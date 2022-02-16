import numpy as np
import matplotlib.pyplot as plt
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

    delta_f_matrix = np.transpose(delta_f_matrix)
    delta_f_matrix = np.nan_to_num(delta_f_matrix)
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







def load_session_data(matlab_file_location, session_name, behaviour_matrix_directory, trial_start, trial_stop, onset_or_offset):

    # Load Matalb Data
    data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

    # Extract Delta F Matrix
    delta_f_matrix = data_object.dF
    delta_f_matrix = np.nan_to_num(delta_f_matrix)
    delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
    delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(behaviour_matrix_directory, session_name + "_preprocessed_basic", "Behaviour_Matrix.npy"), allow_pickle=True)
    print("Behaviour matrix", np.shape(behaviour_matrix))

    # Get Selected Trials
    stable_odour_vis_1 = []
    stable_visual_vis_1 = []
    perfect_transition_vis_1 = []
    imperfect_transition_vis_1 = []
    all_vis_1 = []

    number_of_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(1, number_of_trials):

        # Load Trial Data
        trial_data = behaviour_matrix[trial_index]

        # Is Trial Vis 1
        trial_type = trial_data[1]
        if trial_type == 1:

            all_vis_1.append(trial_index)

            # Is Trial First In Block
            first_in_block = trial_data[9]
            if first_in_block == 1:

                # Is Trial A Miss
                lick_response = trial_data[2]
                if lick_response == 0:

                    # Check If Perfect Transition (Following Trial Is Correct)
                    if behaviour_matrix[trial_index + 1][3] == 1 and behaviour_matrix[trial_index + 2][3] and behaviour_matrix[trial_index + 3][3]:
                        perfect_transition_vis_1.append(trial_index)

                    else:
                        if behaviour_matrix[trial_index + 1][3] == 0 and behaviour_matrix[trial_index + 2][3] and behaviour_matrix[trial_index + 3][3] and behaviour_matrix[trial_index + 4][3]:
                            imperfect_transition_vis_1.append(trial_index)

            else:

                # is trial a hit
                trial_outcome = behaviour_matrix[trial_index][3]
                if trial_outcome == 1:
                    stable_visual_vis_1.append(trial_index)

        # Is Odour
        elif trial_type == 3 or trial_type == 4:

            # irrel stim is vis 1
            irrel_stim = behaviour_matrix[trial_index][6]
            if irrel_stim == 1:

                # Correctly Ignored
                ignore_irrel = behaviour_matrix[trial_index][7]
                if ignore_irrel == 1:

                    # Trial Correct
                    trial_outcome = behaviour_matrix[trial_index][3]
                    if trial_outcome == 1:
                        stable_odour_vis_1.append(trial_index)

    # Get Selected Onsets
    stable_odour_vis_1_onsets = []
    stable_visual_vis_1_onsets = []
    perfect_transition_vis_1_onsets = []
    imperfect_transition_vis_1_onsets = []
    all_vis_1_onsets = []

    for trial in stable_odour_vis_1:

        if onset_or_offset == 'onset':
            stable_odour_vis_1_onsets.append(behaviour_matrix[trial][13])
        if onset_or_offset == 'offset':
            stable_odour_vis_1_onsets.append(behaviour_matrix[trial][14])

    for trial in perfect_transition_vis_1:

        if onset_or_offset == 'onset':
            perfect_transition_vis_1_onsets.append(behaviour_matrix[trial][11])
        if onset_or_offset == 'offset':
            perfect_transition_vis_1_onsets.append(behaviour_matrix[trial][12])

    for trial in imperfect_transition_vis_1:

        if onset_or_offset == 'onset':
            imperfect_transition_vis_1_onsets.append(behaviour_matrix[trial][11])
        if onset_or_offset == 'offset':
            imperfect_transition_vis_1_onsets.append(behaviour_matrix[trial][12])

    for trial in stable_visual_vis_1:

        if onset_or_offset == 'onset':
            stable_visual_vis_1_onsets.append(behaviour_matrix[trial][11])
        if onset_or_offset == 'offset':
            stable_visual_vis_1_onsets.append(behaviour_matrix[trial][12])

    for trial in all_vis_1:

        if onset_or_offset == 'onset':
            all_vis_1_onsets.append(behaviour_matrix[trial][11])
        if onset_or_offset == 'offset':
            all_vis_1_onsets.append(behaviour_matrix[trial][12])

    # Combine Into All Onsets
    all_onsets = stable_odour_vis_1_onsets + perfect_transition_vis_1_onsets + imperfect_transition_vis_1_onsets + stable_visual_vis_1_onsets
    all_onsets.sort()

    # Remove Trials Too Near End
    number_of_timepoints = np.shape(delta_f_matrix)[1]
    valid_onsets = []
    for onset in all_onsets:
        if onset + trial_stop < number_of_timepoints:
            valid_onsets.append(onset)

    print("All onsets length", len(valid_onsets))

    # Get Tensors
    full_tensor = create_trial_tensor(delta_f_matrix, valid_onsets, trial_start, trial_stop)

    print("Full tensor shape", np.shape(full_tensor))

    # Get Indicies Within All Onets
    stable_visual_indicies = []
    stable_odour_indicies = []
    perfect_transition_indicies = []
    imperfect_transition_indicies = []

    for onset in stable_visual_vis_1_onsets:
        if onset in valid_onsets:
            stable_visual_indicies.append(valid_onsets.index(onset))

    for onset in stable_odour_vis_1_onsets:
        if onset in valid_onsets:
            stable_odour_indicies.append(valid_onsets.index(onset))

    for onset in perfect_transition_vis_1_onsets:
        if onset in valid_onsets:
            perfect_transition_indicies.append(valid_onsets.index(onset))

    for onset in imperfect_transition_vis_1_onsets:
        if onset in valid_onsets:
            imperfect_transition_indicies.append(valid_onsets.index(onset))


    print("All Onsets", len(valid_onsets))
    print("Stable Visual Indicies", len(stable_visual_indicies))
    print("Stable Odour Indicies", len(stable_odour_indicies))
    print("Perfect Transition Indicies", len(perfect_transition_indicies))
    print("Imperfect Transition Indicies", len(imperfect_transition_indicies))
    print("FUll Tensor Shape", np.shape(full_tensor))
    return full_tensor, stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies, all_vis_1_onsets, perfect_transition_vis_1_onsets, imperfect_transition_vis_1_onsets





def load_data_for_tca(matlab_file_location, behaviour_matrix_directory, trial_start, trial_stop, onset_or_offset):

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

    # Get Onsets
    full_tensor, stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies, all_vis_1_onsets, perfect_transition_onsets, imperfect_transition_onsets = load_session_data(matlab_file_location, session_name, behaviour_matrix_directory, trial_start, trial_stop, onset_or_offset)

    return full_tensor, stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies, all_vis_1_onsets, perfect_transition_onsets, delta_f_matrix, imperfect_transition_onsets