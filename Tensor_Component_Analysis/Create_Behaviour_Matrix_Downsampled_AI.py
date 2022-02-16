import math

import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec, patches


sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
sys.path.append("/home/matthew/Documents/Github_Code/Behaviour_Analysis")

#import Widefield_General_Functions
import Get_Stable_Windows
import Plot_Behaviour_Matrix
import Import_Preprocessed_Data


def get_ai_filename(base_directory):

    #Get List of all files
    file_list = os.listdir(base_directory)
    ai_filename = None

    #Get .h5 files
    h5_file_list = []
    for file in file_list:
        if file[-3:] == ".h5":
            h5_file_list.append(file)

    #File the H5 file which is two dates seperated by a dash
    for h5_file in h5_file_list:
        original_filename = h5_file

        #Remove Ending
        h5_file = h5_file[0:-3]

        #Split By Dashes
        h5_file = h5_file.split("-")

        if len(h5_file) == 2 and h5_file[0].isnumeric() and h5_file[1].isnumeric():
            ai_filename = "/" + original_filename
            print("Ai filename is: ", ai_filename)
            return ai_filename




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




def get_step_onsets(trace, threshold=1, window=10):
    state = 0
    number_of_timepoints = len(trace)
    onset_times = []
    time_below_threshold = 0

    onset_line = []

    for timepoint in range(number_of_timepoints):
        if state == 0:
            if trace[timepoint] > threshold:
                state = 1
                onset_times.append(timepoint)
                time_below_threshold = 0
            else:
                pass
        elif state == 1:
            if trace[timepoint] > threshold:
                time_below_threshold = 0
            else:
                time_below_threshold += 1
                if time_below_threshold > window:
                    state = 0
                    time_below_threshold = 0
        onset_line.append(state)

    return onset_times



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



def split_stream_by_context(stimuli_onsets, context_onsets, context_window):
    context_negative_onsets = []
    context_positive_onsets = []

    # Iterate Through Visual 1 Onsets
    for stimuli_onset in stimuli_onsets:
        context = False
        window_start = stimuli_onset
        window_end = stimuli_onset + context_window

        for context_onset in context_onsets:
            if context_onset >= window_start and context_onset <= window_end:
                context = True

        if context == True:
            context_positive_onsets.append(stimuli_onset)
        else:
            context_negative_onsets.append(stimuli_onset)

    return context_negative_onsets, context_positive_onsets



def split_visual_onsets_by_context(visual_1_onsets, visual_2_onsets, odour_1_onsets, odour_2_onsets, following_window_size=30):

    combined_odour_onsets = odour_1_onsets + odour_2_onsets
    visual_block_stimuli_1, odour_block_stimuli_1 = split_stream_by_context(visual_1_onsets, combined_odour_onsets, following_window_size)
    visual_block_stimuli_2, odour_block_stimuli_2 = split_stream_by_context(visual_2_onsets, combined_odour_onsets, following_window_size)

    onsets_list = [visual_block_stimuli_1, visual_block_stimuli_2, odour_block_stimuli_1, odour_block_stimuli_2]

    return onsets_list


def get_offset(onset, stream, threshold=0.5):

    count = 0
    on = True
    while on:
        if stream[onset + count] < threshold and count > 10:
            on = False
            return onset + count
        else:
            count += 1




def get_frame_indexes(frame_stream):
    frame_indexes = {}
    state = 1
    threshold = 2
    count = 0

    for timepoint in range(0, len(frame_stream)):

        if frame_stream[timepoint] > threshold:
            if state == 0:
                state = 1
                frame_indexes[timepoint] = count
                count += 1

        else:
            if state == 1:
                state = 0
            else:
                pass

    return frame_indexes






def extract_onsets(ai_data, save_directory, lick_threshold=0.13):


    # Create Stimuli Dictionary
    stimuli_dictionary = create_stimuli_dictionary()

    # Load Traces
    lick_trace = ai_data[stimuli_dictionary["Lick"]]
    vis_1_trace = ai_data[stimuli_dictionary["Visual 1"]]
    vis_2_trace = ai_data[stimuli_dictionary["Visual 2"]]
    odour_1_trace = ai_data[stimuli_dictionary["Odour 1"]]
    odour_2_trace = ai_data[stimuli_dictionary["Odour 2"]]
    reward_trace = ai_data[stimuli_dictionary["Reward"]]
    relevance_trace = ai_data[stimuli_dictionary["Irrelevance"]]
    end_trace = ai_data[stimuli_dictionary["Trial End"]]

    # Create Some Combined Traces
    combined_odour_trace  = np.max([odour_1_trace, odour_2_trace], axis=0)
    combined_visual_trace = np.max([vis_1_trace,   vis_2_trace],   axis=0)

    # Get Onsets
    vis_1_onsets    = get_step_onsets(vis_1_trace)
    vis_2_onsets    = get_step_onsets(vis_2_trace)
    odour_1_onsets  = get_step_onsets(odour_1_trace)
    odour_2_onsets  = get_step_onsets(odour_2_trace)
    lick_onsets     = get_step_onsets(lick_trace, threshold=lick_threshold, window=10)
    reward_onsets   = get_step_onsets(reward_trace)
    end_onsets      = get_step_onsets(end_trace)

    # Split Visual Onsets By Context
    visual_onsets_by_context = split_visual_onsets_by_context(vis_1_onsets, vis_2_onsets, odour_1_onsets, odour_2_onsets)
    vis_context_vis_1_onsets    = visual_onsets_by_context[0]
    vis_context_vis_2_onsets    = visual_onsets_by_context[1]
    odour_context_vis_1_onsets  = visual_onsets_by_context[2]
    odour_context_vis_2_onsets  = visual_onsets_by_context[3]

    onsets_dictionary ={"vis_1_onsets":vis_1_onsets,
                        "vis_2_onsets":vis_2_onsets,
                        "odour_1_onsets":odour_1_onsets,
                        "odour_2_onsets":odour_2_onsets,
                        "lick_onsets":lick_onsets,
                        "reward_onsets":reward_onsets,
                        "vis_context_vis_1_onsets":vis_context_vis_1_onsets,
                        "vis_context_vis_2_onsets":vis_context_vis_2_onsets,
                        "odour_context_vis_1_onsets":odour_context_vis_1_onsets,
                        "odour_context_vis_2_onsets":odour_context_vis_2_onsets,
                        "trial_ends":end_onsets}

    traces_dictionary ={"lick_trace":lick_trace,
                        "vis_1_trace":vis_1_trace,
                        "vis_2_trace":vis_2_trace,
                        "odour_1_trace":odour_1_trace,
                        "odour_2_trace":odour_2_trace,
                        "reward_trace":reward_trace,
                        "relevance_trace":relevance_trace,
                        "combined_odour_trace":combined_odour_trace,
                        "combined_visual_trace":combined_visual_trace,
                        "end_trace":end_trace}

    return onsets_dictionary, traces_dictionary



def get_trial_type(onset, onsets_dictionary):
    vis_context_vis_1_onsets = onsets_dictionary["vis_context_vis_1_onsets"]
    vis_context_vis_2_onsets = onsets_dictionary["vis_context_vis_2_onsets"]
    odour_1_onsets = onsets_dictionary["odour_1_onsets"]
    odour_2_onsets = onsets_dictionary["odour_2_onsets"]

    if onset in vis_context_vis_1_onsets:
        return 1
    elif onset in vis_context_vis_2_onsets:
        return 2
    elif onset in odour_1_onsets:
        return 3
    elif onset in odour_2_onsets:
        return  4


def get_trial_end(onset, onsets_dictionary, traces_dictionary):

    # If Not End - AI May Have Stopped Prematurely - Trial End Will Be Last Part of AI Recorder
    ends_trace = traces_dictionary['end_trace']

    trial_ends = onsets_dictionary["trial_ends"]
    trial_ends.sort()

    for end in trial_ends:
        if end > onset:
            return end
    return len(ends_trace)


def get_stimuli_offset(onset, trial_type, traces_dictionary):

    if trial_type == 1:
        stream = traces_dictionary['vis_1_trace']
    elif trial_type == 2:
        stream = traces_dictionary['vis_2_trace']
    elif trial_type == 3:
        stream = traces_dictionary['odour_1_trace']
    elif trial_type == 4:
        stream = traces_dictionary['odour_2_trace']

    offset = get_offset(onset, stream)
    return offset


def check_lick(onset, offset, traces_dictionary, lick_threshold):

    # Get Lick Trace
    lick_trace = traces_dictionary['lick_trace']

    # Get Lick Trace For Trial
    trial_lick_trace = lick_trace[onset:offset]

    if np.max(trial_lick_trace) >= lick_threshold:
        return 1
    else:
        return 0


def check_reward_outcome(onset, trial_end, traces_dictionary):

    reward_trace = traces_dictionary['reward_trace']

    trial_reward_trace = reward_trace[onset:trial_end]

    if np.max(trial_reward_trace > 0.5):
        return  1
    else:
        return 0


def get_irrel_details(onset, trial_type, onsets_dictionary, traces_dictionary, irrel_preceeding_window=15):

    # Get Irrel Offsets:
    odour_context_vis_1_onsets = onsets_dictionary["odour_context_vis_1_onsets"]
    odour_context_vis_2_onsets = onsets_dictionary["odour_context_vis_2_onsets"]
    vis_1_trace = traces_dictionary['vis_1_trace']
    vis_2_trace = traces_dictionary['vis_2_trace']

    vis_1_irrel_offsets = []
    for irrel_vis_1_onset in odour_context_vis_1_onsets:
        vis_1_irrel_offsets.append(get_offset(irrel_vis_1_onset, vis_1_trace))

    vis_2_irrel_offsets = []
    for irrel_vis_2_onset in odour_context_vis_2_onsets:
        vis_2_irrel_offsets.append(get_offset(irrel_vis_2_onset, vis_2_trace))


    preceeded = 0
    irrel_type = np.nan
    irrel_onset = np.nan

    if trial_type == 1 or trial_type == 2:
        return preceeded, irrel_type, irrel_onset

    else:
        window_start = (onset - irrel_preceeding_window)
        window_stop = onset

        #print("Onset", onset)
        #print("Window start", window_start)
        #print("Window Stop", window_stop)

        irrel_trial_index = 0
        for candidate_irrel_offset in vis_1_irrel_offsets:
            if candidate_irrel_offset > window_start and candidate_irrel_offset < window_stop:
                preceeded = 1
                irrel_type = 1
                irrel_onset = odour_context_vis_1_onsets[irrel_trial_index]
                return  preceeded, irrel_type, irrel_onset
            irrel_trial_index += 1

        irrel_trial_index = 0
        for candidate_irrel_offset in vis_2_irrel_offsets:
            if candidate_irrel_offset > window_start and candidate_irrel_offset < window_stop:
                preceeded = 1
                irrel_type = 2
                irrel_onset = odour_context_vis_2_onsets[irrel_trial_index]
                return  preceeded, irrel_type, irrel_onset
            irrel_trial_index += 1

        return preceeded, irrel_type, irrel_onset




def get_irrel_offset(irrel_onset, irrel_type, traces_dictionary):

    if math.isnan(irrel_type):
        return np.nan

    elif irrel_type == 1:
        irrel_trace = traces_dictionary["vis_1_trace"]

    elif irrel_type == 2:
        irrel_trace = traces_dictionary["vis_2_trace"]

    offset = get_offset(irrel_onset, irrel_trace)

    return offset


def get_ignore_irrel(irrel_onset, irrel_offset, traces_dictionary, lick_threshold):

    if math.isnan(irrel_onset) or math.isnan(irrel_offset):
        return np.nan
    else:
        lick_trace = traces_dictionary['lick_trace']
        irrel_lick_trace = lick_trace[irrel_onset:irrel_offset]
        if np.max(irrel_lick_trace) >= lick_threshold:
            return 0
        else:
            return 1


def check_correct(trial_type, lick):

    if trial_type == 1 or trial_type == 3:
        if lick == 1:
            return 1
        else:
            return 0

    elif trial_type == 2 or trial_type == 4:
        if lick == 0:
            return 1
        else:
            return 0



def classify_trial(onset, onsets_dictionary, traces_dictionary, trial_index, lick_threshold):

    """
    0 trial_index = int, index of trial
    1 trial_type = 1 - rewarded visual, 2 - unrewarded visual, 3 - rewarded odour, 4 - unrewarded odour
    2 lick = 1- lick, 0 - no lick
    3 correct = 1 - correct, 0 - incorrect
    4 rewarded = 1- yes, 0 - no
    5 preeceded_by_irrel = 0 - no, 1 - yes
    6 irrel_type = 1 - rewarded grating, 2 - unrearded grating
    7 ignore_irrel = 0 - licked to irrel, 1 - ignored irrel, nan - no irrel,
    8 block_number = int, index of block
    9 first_in_block = 1 - yes, 2- no
    10 in_block_of_stable_performance = 1 - yes, 2 - no
    11 onset = float onset of major stimuli
    12 stimuli_offset = float offset of major stimuli
    13 irrel_onset = float onset of any irrel stimuli, nan = no irrel stimuli
    14 irrel_offset = float offset of any irrel stimuli, nan = no irrel stimuli
    15 trial_end = float end of trial
    16 Photodiode Onset = Adjusted Visual stimuli onset to when the photodiode detects the stimulus
    17 Photodiode Offset = Adjusted Visual Stimuli Offset to when the photodiode detects the stimulus
    """

    # Get Trial Type
    trial_type = get_trial_type(onset, onsets_dictionary)

    # Get Trial End
    trial_end = get_trial_end(onset, onsets_dictionary, traces_dictionary)

    # Get Stimuli Offset
    stimuli_offset = get_stimuli_offset(onset, trial_type, traces_dictionary)

    # Get Mouse Response
    lick = check_lick(onset, trial_end, traces_dictionary, lick_threshold)

    # Check Correct
    correct = check_correct(trial_type, lick)

    # Check Reward Outcome
    rewarded = check_reward_outcome(onset, trial_end, traces_dictionary)

    # Get Irrel Details
    preeceded_by_irrel, irrel_type, irrel_onset = get_irrel_details(onset, trial_type, onsets_dictionary, traces_dictionary)

    # Get Irrel Offset
    irrel_offset = get_irrel_offset(irrel_onset, irrel_type, traces_dictionary)

    # Get Ignore Irrel
    ignore_irrel = get_ignore_irrel(irrel_onset, irrel_offset, traces_dictionary, lick_threshold)

    first_in_block = None
    in_block_of_stable_performance = 0
    block_number = None


    trial_vector = [trial_index,
                    trial_type,
                    lick,
                    correct,
                    rewarded,
                    preeceded_by_irrel,
                    irrel_type,
                    ignore_irrel,
                    block_number,
                    first_in_block,
                    in_block_of_stable_performance,
                    onset,
                    stimuli_offset,
                    irrel_onset,
                    irrel_offset,
                    trial_end]

    return trial_vector



def print_behaviour_matrix(behaviour_matrix):

    for t in behaviour_matrix:
        print("Trial:",t[0],"Type:",t[1],"Lick:",t[2],"Correct:",t[3],"Rewarded:",t[4],"Irrel_Preceed:",t[5],"Irrel Type:",t[6],"Ignore Irrel:",t[7],"Block Number:",t[8],"First In Block:",t[9],"In Stable Window:",t[10],"Onset:",t[11],"Offset:",t[12],"Irrel Onset",t[13],"Irrel Offset",t[14])





def get_block_boudaries(onsets_dictionary):

    odour_1_onsets              = onsets_dictionary["odour_1_onsets"]
    odour_2_onsets              = onsets_dictionary["odour_2_onsets"]
    vis_context_vis_1_onsets    = onsets_dictionary["vis_context_vis_1_onsets"]
    vis_context_vis_2_onsets    = onsets_dictionary["vis_context_vis_2_onsets"]

    # Get Visual trial Stimuli,
    vis_context_stimuli = np.concatenate([vis_context_vis_1_onsets, vis_context_vis_2_onsets])
    vis_context_stimuli.sort()

    # Get Odour Trial Stimuli
    odour_context_stimuli = np.concatenate([odour_1_onsets, odour_2_onsets])
    odour_context_stimuli.sort()

    all_onsets = np.concatenate([vis_context_stimuli, odour_context_stimuli])
    all_onsets = np.sort(all_onsets)

    block_boundaries = [0]
    block_types = []

    # Get Initial Block
    if vis_context_stimuli[0] < odour_context_stimuli[0]:
        initial_block = 0
    else:
        initial_block = 1
    block_types.append(initial_block)

    # Get Subsequent Blocks
    current_block = initial_block

    number_of_trials = len(all_onsets)
    for trial in range(1, number_of_trials):

        onset = all_onsets[trial]

        # If Its a Visual Onset
        if onset in vis_context_stimuli:
            if current_block == 1:
                current_block = 0
                block_boundaries.append(trial)
                block_types.append(current_block)

        elif onset in odour_context_stimuli:
            if current_block == 0:
                current_block = 1
                block_boundaries.append(trial)
                block_types.append(current_block)

    return block_boundaries, block_types


def add_block_boundaries(trial_matrix, block_boundaries):

    # remove first boundary (first trial)
    block_boundaries = block_boundaries[1:]
    current_block = 0

    number_of_trials = np.shape(trial_matrix)[0]
    for trial_index in range(number_of_trials):
        if trial_index in block_boundaries:
            trial_matrix[trial_index][9] = 1
            current_block += 1
        else:
            trial_matrix[trial_index][9] = 0

        trial_matrix[trial_index][8] = current_block

    return trial_matrix


def add_stable_windows(behaviour_matrix, stable_windows):

    for window in stable_windows:
        for trial in window:
            behaviour_matrix[trial][10] = 1
    return behaviour_matrix



def get_nearest_frame(stimuli_onsets, frame_times):

    #frame_times = frame_onsets.keys()
    nearest_frames = []
    window_size = 50

    for onset in stimuli_onsets:
        smallest_distance = 1000
        closest_frame = None

        window_start = int(onset - window_size)
        window_stop  = int(onset + window_size)

        for timepoint in range(window_start, window_stop):

            #There is a frame at this time
            if timepoint in frame_times:
                distance = abs(onset - timepoint)

                if distance < smallest_distance:
                    smallest_distance = distance
                    closest_frame = frame_times.index(timepoint)
                    #closest_frame = frame_onsets[timepoint]

        if closest_frame != None:
            if closest_frame > 11:
                nearest_frames.append(closest_frame)

    nearest_frames = np.array(nearest_frames)
    return nearest_frames



def get_times_from_behaviour_matrix(behaviour_matrix, selected_trials, onset_category):
    trial_times = []
    for trial in selected_trials:
        relevant_onset = behaviour_matrix[trial][onset_category]
        trial_times.append(relevant_onset)
    return trial_times






def save_onsets(behaviour_matrix, selected_trials, save_directory):

    # Load Trials
    visual_context_stable_vis_1_trials      = selected_trials[0]
    visual_context_stable_vis_2_trials      = selected_trials[1]
    odour_context_stable_vis_1_trials       = selected_trials[2]
    odour_context_stable_vis_2_trials       = selected_trials[3]
    perfect_transition_trials               = selected_trials[4]
    odour_expected_present_trials           = selected_trials[5]
    odour_not_expected_not_present_trials   = selected_trials[6]
    odour_1_cued                            = selected_trials[7]
    odour_2_cued                            = selected_trials[8]
    odour_1_not_cued                        = selected_trials[9]
    odour_2_not_cued                        = selected_trials[10]

    # Get Stimuli Times For Each Trial Cateogry
    visual_context_stable_vis_1_times      = get_times_from_behaviour_matrix(behaviour_matrix, visual_context_stable_vis_1_trials,    11) #11
    visual_context_stable_vis_2_times      = get_times_from_behaviour_matrix(behaviour_matrix, visual_context_stable_vis_2_trials,    11) #11
    odour_context_stable_vis_1_times       = get_times_from_behaviour_matrix(behaviour_matrix, odour_context_stable_vis_1_trials,     13) #13
    odour_context_stable_vis_2_times       = get_times_from_behaviour_matrix(behaviour_matrix, odour_context_stable_vis_2_trials,     13) #13
    perfect_transition_times               = get_times_from_behaviour_matrix(behaviour_matrix, perfect_transition_trials,             12) #12
    odour_expected_present_times           = get_times_from_behaviour_matrix(behaviour_matrix, odour_expected_present_trials,         14) #14
    odour_not_expected_not_present_times   = get_times_from_behaviour_matrix(behaviour_matrix, odour_not_expected_not_present_trials, 12) #12
    odour_1_cued_times                     = get_times_from_behaviour_matrix(behaviour_matrix, odour_1_cued,                          11) #11
    odour_2_cued_times                     = get_times_from_behaviour_matrix(behaviour_matrix, odour_2_cued,                          11) #11
    odour_1_not_cued_times                 = get_times_from_behaviour_matrix(behaviour_matrix, odour_1_not_cued,                      11) #11
    odour_2_not_cued_times                 = get_times_from_behaviour_matrix(behaviour_matrix, odour_2_not_cued,                      11) #11

    # Get Frames For Each Stimuli Category
    visual_context_stable_vis_1_onsets      = visual_context_stable_vis_1_times
    visual_context_stable_vis_2_onsets      = visual_context_stable_vis_2_times
    odour_context_stable_vis_1_onsets       = odour_context_stable_vis_1_times
    odour_context_stable_vis_2_onsets       = odour_context_stable_vis_2_times
    perfect_transition_onsets               = perfect_transition_times
    odour_expected_present_onsets           = odour_expected_present_times
    odour_not_expected_not_present_onsets   = odour_not_expected_not_present_times
    odour_1_cued_onsets                     = odour_1_cued_times
    odour_2_cued_onsets                     = odour_2_cued_times
    odour_1_not_cued_onsets                 = odour_1_not_cued_times
    odour_2_not_cued_onsets                 = odour_2_not_cued_times

    # Save Onsets
    np.save(os.path.join(save_directory, "visual_context_stable_vis_1_onsets.npy"),     visual_context_stable_vis_1_onsets)
    np.save(os.path.join(save_directory, "visual_context_stable_vis_2_onsets.npy"),     visual_context_stable_vis_2_onsets)
    np.save(os.path.join(save_directory, "odour_context_stable_vis_1_onsets.npy"),      odour_context_stable_vis_1_onsets)
    np.save(os.path.join(save_directory, "odour_context_stable_vis_2_onsets.npy"),      odour_context_stable_vis_2_onsets)
    np.save(os.path.join(save_directory, "perfect_transition_onsets.npy"),              perfect_transition_onsets)
    np.save(os.path.join(save_directory, "odour_expected_present_onsets.npy"),          odour_expected_present_onsets)
    np.save(os.path.join(save_directory, "odour_not_expected_not_present_onsets.npy"),  odour_not_expected_not_present_onsets)
    np.save(os.path.join(save_directory, "odour_1_cued_onsets.npy"),                    odour_1_cued_onsets)
    np.save(os.path.join(save_directory, "odour_2_cued_onsets.npy"),                    odour_2_cued_onsets)
    np.save(os.path.join(save_directory, "odour_1_not_cued_onsets.npy"),                odour_1_not_cued_onsets)
    np.save(os.path.join(save_directory, "odour_2_not_cued_onsets.npy"),                odour_2_not_cued_onsets)
    np.save(os.path.join(save_directory, "Behaviour_Matrix.npy"), behaviour_matrix)


def get_selected_trials(behaviour_matrix):

    """
    Stable Trials
    Visual_Context_Vis_1_Stable - correct, in stable block, not first in block
    Odour_Context_Vis_2_Stable - correct, in stable block, not first in block
    Visual_Context_Vis_1_Stable - correct, in stable block, not first in block, ignored irrel
    Odour_Context_Vis_2_Stable - correct, in stable block, not first in block, ignored irrel


    Absence of Expected Odour
    Perfect Switch Trials  – first in visual block, vis 1 irrel, miss, next trial correct
    Odour_Expected_Present – Odour 2, correct, preceeded by irrel, ignore irrel
    Odour_Not_Expected_Absent – visual block, end of vis 2, correct,


    Cued v Non-Cued Odour
    odour_1_cued - Odour 1 - correct - in stable block - preceeded by irrel
    odour_2_cued - Odour 2 - correct - in stable block - preceeded by irrel
    odour_1_not_cued - Odour 2 - correct - in stable block - not preceeded by irrel
    odour_2_not_cued - Odour 2 - correct - in stable block - not preceeded by irrel
    """

    # Get Selected Trials
    visual_context_stable_vis_1_trials = []
    visual_context_stable_vis_2_trials = []
    odour_context_stable_vis_1_trials = []
    odour_context_stable_vis_2_trials = []

    perfect_transition_trials = []
    odour_expected_present_trials = []
    odour_not_expected_not_present_trials = []

    odour_1_cued = []
    odour_2_cued = []
    odour_1_not_cued = []
    odour_2_not_cued = []

    number_of_trials = np.shape(behaviour_matrix)[0]

    for trial_index in range(number_of_trials):

        trial_is_correct    = behaviour_matrix[trial_index][3]
        in_stable_window    = behaviour_matrix[trial_index][10]
        trial_type          = behaviour_matrix[trial_index][1]
        first_in_block      = behaviour_matrix[trial_index][9]
        ignore_irrel        = behaviour_matrix[trial_index][7]
        preeceeded_by_irrel = behaviour_matrix[trial_index][5]

        # Check If Trial Is Stable
        if trial_is_correct and ignore_irrel and in_stable_window and not first_in_block:

            if trial_type == 1:
                visual_context_stable_vis_1_trials.append(trial_index)
            elif trial_type == 2:
                visual_context_stable_vis_2_trials.append(trial_index)

            elif trial_type == 3 or trial_type == 4:
                irrel_type = behaviour_matrix[trial_index][6]

                if irrel_type == 1:
                    odour_context_stable_vis_1_trials.append(trial_index)
                elif irrel_type == 2:
                    odour_context_stable_vis_2_trials.append(trial_index)

        # Check If Trial is A Perfect Transition Trial
        # visual block - vi 1
        # first in block
        # miss
        # next trial correct

        if trial_type == 1 and first_in_block:
            following_trial_correct = behaviour_matrix[trial_index + 1][3]
            if not trial_is_correct and following_trial_correct:
                perfect_transition_trials.append(trial_index)

        # Check If Is Odour Expected, Present
        # Odour 2, correct, stable, preceeded by irrel, ignore irrel, not first in block
        if trial_type == 4 and trial_is_correct and in_stable_window and preeceeded_by_irrel and ignore_irrel and not first_in_block:
            odour_expected_present_trials.append(trial_index)

        # Check If odour_not_expected_not_present_trials
        # visual block - vis 2, correct,
        if trial_type == 2 and trial_is_correct:
            odour_not_expected_not_present_trials.append(trial_index)

        # Check If Cued
        if trial_type == 3 or trial_type == 4 and trial_is_correct and ignore_irrel and in_stable_window:

            if trial_type == 3 and preeceeded_by_irrel:
                odour_1_cued.append(trial_index)

            if trial_type == 4 and preeceeded_by_irrel:
                odour_2_cued.append(trial_index)

            if trial_type == 3 and not preeceeded_by_irrel:
                odour_1_not_cued.append(trial_index)

            if trial_type == 4 and not preeceeded_by_irrel:
                odour_2_not_cued.append(trial_index)

    selected_trials_list = [
                            visual_context_stable_vis_1_trials,
                            visual_context_stable_vis_2_trials,
                            odour_context_stable_vis_1_trials,
                            odour_context_stable_vis_2_trials,

                            perfect_transition_trials,
                            odour_expected_present_trials,
                            odour_not_expected_not_present_trials,

                            odour_1_cued,
                            odour_2_cued,
                            odour_1_not_cued,
                            odour_2_not_cued]

    return selected_trials_list



def get_step_onsets_photodiode(trace, threshold=1, window=10):

    state = 0
    number_of_timepoints = len(trace)
    onset_times = []
    time_below_threshold = 0

    onset_line = []

    for timepoint in range(number_of_timepoints-window):
        if state == 0:
            if trace[timepoint] > threshold and trace[timepoint+window] > threshold:
                state = 1
                onset_times.append(timepoint)
                time_below_threshold = 0
            else:
                pass
        elif state == 1:
            if trace[timepoint] > threshold:
                time_below_threshold = 0
            else:
                time_below_threshold += 1
                if time_below_threshold > window:
                    state = 0
                    time_below_threshold = 0
        onset_line.append(state)

    return onset_times, onset_line



def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list


def check_lick_threshold(ai_data, lick_threshold):

    # Create Stimuli Dictionary
    stimuli_dictionary = create_stimuli_dictionary()

    # Get Lick Trace
    lick_trace = ai_data[stimuli_dictionary['Lick']]

    # Plot Lick Trace And Threshold
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.plot(lick_trace)
    axis_1.axhline(lick_threshold, c='k')
    plt.show()



def create_save_directory(save_directory):

    if not os.path.isdir(save_directory):
        print("Making save directory", save_directory)
        os.mkdir(save_directory)


def create_behaviour_matrix(matlab_file_location, save_directory, lick_threshold=0.12):

    # Create Save Directory
    create_save_directory(save_directory)

    # Load AI Data
    ai_data = load_ai_recorder_file(matlab_file_location)

    # Check Lick Threshold
    check_lick_threshold(ai_data, lick_threshold)

    # Get Trace and Onsets Dictionary
    onsets_dictionary, traces_dictionary = extract_onsets(ai_data, save_directory, lick_threshold=lick_threshold)

    # Create Trial Onsets List
    vis_context_vis_1_onsets = onsets_dictionary["vis_context_vis_1_onsets"]
    vis_context_vis_2_onsets = onsets_dictionary["vis_context_vis_2_onsets"]
    odour_1_onsets = onsets_dictionary["odour_1_onsets"]
    odour_2_onsets = onsets_dictionary["odour_2_onsets"]
    trial_onsets = vis_context_vis_1_onsets + vis_context_vis_2_onsets + odour_1_onsets + odour_2_onsets
    trial_onsets.sort()



    # Classify Trials
    trial_matrix = []
    trial_index = 0
    for trial in trial_onsets:
        trial_vector = classify_trial(trial, onsets_dictionary, traces_dictionary, trial_index, lick_threshold=lick_threshold)
        trial_matrix.append(trial_vector)
        trial_index += 1
    trial_matrix = np.array(trial_matrix)

    # Add Block Boundaries
    block_boundaries, block_types = get_block_boudaries(onsets_dictionary)
    trial_matrix = add_block_boundaries(trial_matrix, block_boundaries)

    # Get Stable Windows
    stable_windows = Get_Stable_Windows.get_stable_windows(trial_matrix)
    trial_matrix = add_stable_windows(trial_matrix, stable_windows)

    # Get Selected Trials
    selected_trials = get_selected_trials(trial_matrix)

    # Print Behaviour Matrix
    print_behaviour_matrix(trial_matrix)

    # Save Trials
    save_onsets(trial_matrix, selected_trials ,save_directory)

    # Plot Behaviour Matrix
    Plot_Behaviour_Matrix.plot_behaviour_maxtrix(save_directory, trial_matrix, onsets_dictionary, block_boundaries, stable_windows, selected_trials)




# Load Matlab Data
base_directory = "/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Best_switching_sessions_all_sites"


file_list = load_matlab_sessions(base_directory)

for matlab_file in file_list:

    session_name = matlab_file.split("/")[-1]
    print("Matalb file", matlab_file)

    save_directory = os.path.join("/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Behaviour_Matricies", session_name[0:-4])
    print("Save directory:", save_directory)
    create_behaviour_matrix(matlab_file, save_directory)
