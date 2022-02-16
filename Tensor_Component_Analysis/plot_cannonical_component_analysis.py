import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist
import jPCA

from mvlearn.datasets import load_UCImultifeature
from mvlearn.embed import GCCA

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)





def view_projection_matricies(gcca_dictionary):
    projection_matrix_list = gcca_dictionary['Projection_Matricies']

    for projection_matrix in projection_matrix_list:

        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1,1,1)

        first_3_components = projection_matrix[:,0:3]

        # Sort Matrix
        transition_component = first_3_components[:, 2]
        sorting_indicies = np.argsort(transition_component)
        sorting_indicies = np.flip(sorting_indicies)

        print("Transition Component", transition_component)
        print("Sorting Indicies", sorting_indicies)

        sorted_3_components = []
        for index in sorting_indicies:
            sorted_3_components.append(first_3_components[index])

        sorted_3_components = np.array(sorted_3_components)
        print("Sorted Components SHape", np.shape(sorted_3_components))




        loading_range = np.max(np.abs(first_3_components))

        axis_1.imshow(sorted_3_components, cmap='seismic', vmin=-1*loading_range, vmax=loading_range)
        forceAspect(axis_1)

        plt.show()




def plot_factor_decay(gcca_dictionary):


    all_vis_1_onsets_list = gcca_dictionary['All_Vis_1_Onsets']
    perfect_transition_onsets_list = gcca_dictionary['Perfect_Transition_Onsets']

    delta_f_matrix_list = gcca_dictionary['Delta_F_Matrix']
    projection_matrix_list = gcca_dictionary['Projection_Matricies']
    trial_start = gcca_dictionary["Trial_Start"]
    trial_stop = gcca_dictionary["Trial_Stop"]

    number_of_sessions = len(all_vis_1_onsets_list)

    mean_trace_list = []
    for session_index in range(number_of_sessions):

        all_vis_1_onsets = all_vis_1_onsets_list[session_index]
        perfect_transition_onsets = perfect_transition_onsets_list[session_index]
        delta_f_matrix = delta_f_matrix_list[session_index]

        number_of_perfect_transitions = len(perfect_transition_onsets)
        print("Delta F Matrix Shape", np.shape(delta_f_matrix))
        print("Number of perfect Transitions", len(perfect_transition_onsets))

        all_vis_1_onsets.sort()
        perfect_transition_onsets.sort()

        # Get Triplet Onsets
        following_trial_list = []
        for onset in perfect_transition_onsets:
            onset_index = all_vis_1_onsets.index(onset)
            following_trial_onsets = all_vis_1_onsets[onset_index: onset_index + 4]
            following_trial_list.append(following_trial_onsets)


        # Get Delta F Tensor
        delta_f_tensor = []
        number_of_transitions = len(perfect_transition_onsets)
        for transition_index in range(number_of_transitions):
            onset_triplet = following_trial_list[transition_index]

            triplet_tensor = []
            for onset in onset_triplet:
                delta_f_trace = delta_f_matrix[:, onset+trial_start: onset+trial_stop]
                delta_f_trace = np.transpose(delta_f_trace)
                triplet_tensor.append(delta_f_trace)

            triplet_tensor = np.array(triplet_tensor)
            triplet_tensor = np.reshape(triplet_tensor,  (np.shape(triplet_tensor)[0] * np.shape(triplet_tensor)[1], np.shape(triplet_tensor)[2]))
            delta_f_tensor.append(triplet_tensor)

        delta_f_tensor = np.array(delta_f_tensor)
        print("Delta F Tensor Shape", np.shape(delta_f_tensor))

        # Convert To Factor 2
        projeciton_matrix = projection_matrix_list[session_index]
        transition_component = projeciton_matrix[:, 2]
        perfect_transition_component_traces = []
        for perfect_transition_index in range(number_of_perfect_transitions):
            delta_f_trace = delta_f_tensor[perfect_transition_index]
            component_trace = np.dot(delta_f_trace, transition_component)
            perfect_transition_component_traces.append(component_trace)

        # Get Mean Trace
        if len(perfect_transition_component_traces) < 2:
            mean_trace = perfect_transition_component_traces[0]
        else:
            perfect_transition_component_traces = np.array(perfect_transition_component_traces)
            mean_trace = np.mean(perfect_transition_component_traces, axis=0)

        mean_trace_list.append(mean_trace)

    for trace in mean_trace_list:
        plt.plot(trace)
    plt.show()


    mean_trace_list = np.array(mean_trace_list)
    mean_mean_trace = np.mean(mean_trace_list, axis=0)
    mean_trace_sd = np.std(mean_trace_list, axis=0)

    plt.plot(mean_mean_trace, c='k')

    x_values = list(range(len(mean_mean_trace)))
    plt.fill_between(x=x_values, y1=mean_mean_trace, y2=np.add(mean_mean_trace, mean_trace_sd), color='b', alpha=0.3)
    plt.fill_between(x=x_values, y1=mean_mean_trace, y2=np.subtract(mean_mean_trace, mean_trace_sd), color='b', alpha=0.3)
    plt.show()



def get_activity_tensor(onset_list, following_trial_list, trial_start, trial_stop, delta_f_matrix):

    delta_f_tensor = []
    number_of_transitions = len(onset_list)
    for transition_index in range(number_of_transitions):
        onset_triplet = following_trial_list[transition_index]

        triplet_tensor = []
        for onset in onset_triplet:
            delta_f_trace = delta_f_matrix[:, onset + trial_start: onset + trial_stop]
            delta_f_trace = np.transpose(delta_f_trace)
            triplet_tensor.append(delta_f_trace)

        triplet_tensor = np.array(triplet_tensor)
        triplet_tensor = np.reshape(triplet_tensor, (np.shape(triplet_tensor)[0] * np.shape(triplet_tensor)[1], np.shape(triplet_tensor)[2]))
        delta_f_tensor.append(triplet_tensor)

    delta_f_tensor = np.array(delta_f_tensor)
    print("Delta F Tensor Shape", np.shape(delta_f_tensor))
    return delta_f_tensor


def project_tensor(tensor, projection_axis):

    number_of_trials = np.shape(tensor)[0]

    component_traces = []
    for trial_index in range(number_of_trials):
        delta_f_trace = tensor[trial_index]
        component_trace = np.dot(delta_f_trace, projection_axis)
        component_traces.append(component_trace)

    return component_traces


def plot_factor_decay_perfect_v_imperfect(gcca_dictionary):

    all_vis_1_onsets_list = gcca_dictionary['All_Vis_1_Onsets']
    perfect_transition_onsets_list = gcca_dictionary['Perfect_Transition_Onsets']
    imperfect_transition_onsets_list = gcca_dictionary['Imperfect_Transition_Onsets']

    print("Imperfect Transitions Onsets List", imperfect_transition_onsets_list)


    delta_f_matrix_list = gcca_dictionary['Delta_F_Matrix']
    projection_matrix_list = gcca_dictionary['Projection_Matricies']
    trial_start = gcca_dictionary["Trial_Start"]
    trial_stop = gcca_dictionary["Trial_Stop"]

    number_of_sessions = len(all_vis_1_onsets_list)

    mean_perfect_trace_list = []
    mean_imperfect_trace_list = []

    for session_index in range(number_of_sessions):

        # Create Lists To Hold Data
        session_perfect_traces = []
        session_imperfect_traces = []

        # Load Onsets
        all_vis_1_onsets = all_vis_1_onsets_list[session_index]
        perfect_transition_onsets = perfect_transition_onsets_list[session_index]
        imperfect_transition_onsets = imperfect_transition_onsets_list[session_index]
        delta_f_matrix = delta_f_matrix_list[session_index]

        number_of_perfect_transitions = len(perfect_transition_onsets)
        number_of_imperfect_transition = len(imperfect_transition_onsets)
        print("Delta F Matrix Shape", np.shape(delta_f_matrix))
        print("Number of perfect Transitions", len(perfect_transition_onsets))
        print("Number of Imperfect Transitions", len(imperfect_transition_onsets))

        all_vis_1_onsets.sort()
        perfect_transition_onsets.sort()
        imperfect_transition_onsets.sort()


        # Get Triplet Onsets
        perfect_following_trial_list = []
        for onset in perfect_transition_onsets:
            onset_index = all_vis_1_onsets.index(onset)
            following_trial_onsets = all_vis_1_onsets[onset_index: onset_index + 4]
            perfect_following_trial_list.append(following_trial_onsets)

        imperfect_following_trial_list = []
        for onset in imperfect_transition_onsets:
            onset_index = all_vis_1_onsets.index(onset)
            following_trial_onsets = all_vis_1_onsets[onset_index: onset_index + 4]
            imperfect_following_trial_list.append(following_trial_onsets)

        # Convert To Delta F
        perfect_following_tensor = get_activity_tensor(perfect_transition_onsets, perfect_following_trial_list, trial_start, trial_stop, delta_f_matrix)
        imperfect_following_tensor = get_activity_tensor(imperfect_transition_onsets, imperfect_following_trial_list, trial_start, trial_stop, delta_f_matrix)

        # Convert To Factor 2
        projeciton_matrix = projection_matrix_list[session_index]
        transition_component = projeciton_matrix[:, 0]

        perfect_component_traces = project_tensor(perfect_following_tensor, transition_component)
        imperfect_component_traces = project_tensor(imperfect_following_tensor, transition_component)

        for trace in perfect_component_traces:
            plt.plot(trace, c='b')

        for trace in imperfect_component_traces:
            plt.plot(trace, c='r')

        plt.show()

    """
        # Get Mean Trace
        if len(perfect_transition_component_traces) < 2:
            mean_trace = perfect_transition_component_traces[0]
        else:
            perfect_transition_component_traces = np.array(perfect_transition_component_traces)
            mean_trace = np.mean(perfect_transition_component_traces, axis=0)

        mean_trace_list.append(mean_trace)

    for trace in mean_trace_list:
        plt.plot(trace)
    plt.show()


    mean_trace_list = np.array(mean_trace_list)
    mean_mean_trace = np.mean(mean_trace_list, axis=0)
    mean_trace_sd = np.std(mean_trace_list, axis=0)

    plt.plot(mean_mean_trace, c='k')

    x_values = list(range(len(mean_mean_trace)))
    plt.fill_between(x=x_values, y1=mean_mean_trace, y2=np.add(mean_mean_trace, mean_trace_sd), color='b', alpha=0.3)
    plt.fill_between(x=x_values, y1=mean_mean_trace, y2=np.subtract(mean_mean_trace, mean_trace_sd), color='b', alpha=0.3)
    plt.show()
    """



def view_trial_averaged_components(gcca_dictionary):

    transformed_trial_averages = gcca_dictionary["Transformed_Trial_Averages"]
    print("Transformed trial averages", np.shape(transformed_trial_averages[0]))

    # Get Data Structure
    number_of_components = np.shape(transformed_trial_averages[0])[1]
    number_of_timepoints = np.shape(transformed_trial_averages[0])[0]
    print("Number of timepoints", number_of_timepoints)
    print("Number of components", number_of_components)

    number_of_sessions = len(transformed_trial_averages)
    trial_start = gcca_dictionary["Trial_Start"]
    trial_stop = gcca_dictionary["Trial_Stop"]
    trial_length = trial_stop - trial_start
    number_of_conditions = int(number_of_timepoints / trial_length)
    print("Number of conditions", number_of_conditions)

    colour_list = ['b', 'g', 'm']

    # Iterate Through Components
    for component_index in range(number_of_components):

        # Create Figure
        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1, 1, 1)
        axis_1.set_title("Component: " + str(component_index))

        # Iterate Through Sessions
        for session_index in range(number_of_sessions):
            axis_1.plot(transformed_trial_averages[session_index][:, component_index])

        #Shade Trial Regions
        for condition_index in range(number_of_conditions):
            start = condition_index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop-1, facecolor=colour_list[condition_index], alpha=0.3)

            axis_1.axvline(x=start-trial_start, ymin=0, ymax=1, c='k')


        plt.show()

    # Shade Trial Regions




    """
    # Plot Full Traces
    for session_index in range(number_of_sessions):


        # Get Full Tensor
        full_tensor = full_tensor_list[session_index]

        # Get Tensor Structure
        number_of_trials = np.shape(full_tensor)[0]
        trial_length = np.shape(full_tensor)[1]
        number_of_neurons = np.shape(full_tensor)[2]
        flat_tensor = np.reshape(full_tensor, (number_of_trials * trial_length, number_of_neurons))
        print("Flat Tensor Shape", np.shape(flat_tensor))
        print("Trial Length", trial_length)

        # Transform Data
        transformed_tensor = np.dot(flat_tensor, projection_matrixies[session_index])
        print("Transformed Tensor Shape", np.shape(transformed_tensor))

        # Get Trial Type Indexes
        trial_type_indexes = trial_type_index_list[session_index]
        visual_indicies = trial_type_indexes[0]
        odour_indicies = trial_type_indexes[1]
        perfect_transition_indicies = trial_type_indexes[2]
        imperfect_transition_indicies = trial_type_indexes[3]

        # Plot Traces
        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1, 1, 1)
        axis_1.plot(transformed_tensor[:, 0], c='b')
        axis_1.plot(transformed_tensor[:, 1], c='g')
        axis_1.plot(transformed_tensor[:, 2], c='m')

        # Fill In Trial Type
        print("Visual Indicies", visual_indicies)
        for index in visual_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop, facecolor='b', alpha=0.3)

        for index in odour_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop, facecolor='g', alpha=0.3)

        for index in perfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop, facecolor='m', alpha=0.3)

        for index in imperfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop, facecolor='orange', alpha=0.3)

        for x in range(0, number_of_trials * trial_length, trial_length):
            axis_1.axvline(x=x, ymin=0, ymax=1, c='k')
        plt.show()
    """

def draw_trial(gcca_dictionary):

    number_of_sessions = len(gcca_dictionary["Session_Names"])
    full_tensor_list = gcca_dictionary["Full_Tensor_List"]
    projection_matrixies = gcca_dictionary["Projection_Matricies"]
    trial_type_index_list = gcca_dictionary["Trial_Type_Indexes"]

    # Plot Full Traces
    for session_index in range(number_of_sessions):

        # Get Full Tensor
        full_tensor = full_tensor_list[session_index]

        # Get Tensor Structure
        number_of_trials = np.shape(full_tensor)[0]
        trial_length = np.shape(full_tensor)[1]
        number_of_neurons = np.shape(full_tensor)[2]
        flat_tensor = np.reshape(full_tensor, (number_of_trials * trial_length, number_of_neurons))
        print("Flat Tensor Shape", np.shape(flat_tensor))
        print("Trial Length", trial_length)

        # Transform Data
        transformed_tensor = np.dot(flat_tensor, projection_matrixies[session_index])
        print("Transformed Tensor Shape", np.shape(transformed_tensor))

        # Get Trial Type Indexes
        trial_type_indexes = trial_type_index_list[session_index]
        visual_indicies = trial_type_indexes[0]
        odour_indicies = trial_type_indexes[1]
        perfect_transition_indicies = trial_type_indexes[2]
        imperfect_transition_indicies = trial_type_indexes[3]

        # Plot Traces
        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1, 1, 1)
        axis_1.plot(transformed_tensor[:, 0], c='b')
        axis_1.plot(transformed_tensor[:, 1], c='g')
        axis_1.plot(transformed_tensor[:, 2], c='m')

        # Fill In Trial Type
        print("Visual Indicies", visual_indicies)
        for index in visual_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop, facecolor='b', alpha=0.3)

        for index in odour_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop, facecolor='g', alpha=0.3)

        for index in perfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop, facecolor='m', alpha=0.3)

        for index in imperfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.axvspan(xmin=start, xmax=stop, facecolor='orange', alpha=0.3)

        for x in range(0, number_of_trials * trial_length, trial_length):
            axis_1.axvline(x=x, ymin=0, ymax=1, c='k')
        plt.show()


def plot_gcca_trajectories(gcca_dictionary):
    number_of_sessions = len(gcca_dictionary["Session_Names"])
    full_tensor_list = gcca_dictionary["Full_Tensor_List"]
    projection_matrixies = gcca_dictionary["Projection_Matricies"]
    trial_type_index_list = gcca_dictionary["Trial_Type_Indexes"]

    # Plot Traces
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1, projection='3d')

    # Plot Full Traces
    for session_index in range(number_of_sessions):

        # Get Full Tensor
        full_tensor = full_tensor_list[session_index]

        # Get Tensor Structure
        number_of_trials = np.shape(full_tensor)[0]
        trial_length = np.shape(full_tensor)[1]
        number_of_neurons = np.shape(full_tensor)[2]
        flat_tensor = np.reshape(full_tensor, (number_of_trials * trial_length, number_of_neurons))
        print("Flat Tensor Shape", np.shape(flat_tensor))
        print("Trial Length", trial_length)

        # Transform Data
        transformed_tensor = np.dot(flat_tensor, projection_matrixies[session_index])
        print("Transformed Tensor Shape", np.shape(transformed_tensor))

        # Get Trial Type Indexes
        trial_type_indexes = trial_type_index_list[session_index]
        visual_indicies = trial_type_indexes[0]
        odour_indicies = trial_type_indexes[1]
        perfect_transition_indicies = trial_type_indexes[2]
        imperfect_transition_indicies = trial_type_indexes[3]

        # Fill In Trial Type

        for index in visual_indicies:
            start = index * trial_length
            stop = start + trial_length
            #axis_1.plot(transformed_tensor[start:stop, 0], transformed_tensor[start:stop, 1], transformed_tensor[start:stop, 2], c='b', alpha=0.5)

        for index in odour_indicies:
            start = index * trial_length
            stop = start + trial_length
            #axis_1.plot(transformed_tensor[start:stop, 0], transformed_tensor[start:stop, 1], transformed_tensor[start:stop, 2], c='g',  alpha=0.5)

        for index in perfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.plot(transformed_tensor[start:stop, 0], transformed_tensor[start:stop, 1], transformed_tensor[start:stop, 2], c='m',  alpha=0.5)

        for index in imperfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.plot(transformed_tensor[start:stop, 0], transformed_tensor[start:stop, 1], transformed_tensor[start:stop, 2], c='orange',  alpha=0.5)


    plt.show()

def get_mean_waveforms(gcca_dictionary):

    number_of_sessions = len(gcca_dictionary["Session_Names"])
    full_tensor_list = gcca_dictionary["Full_Tensor_List"]
    projection_matrixies = gcca_dictionary["Projection_Matricies"]
    trial_type_index_list = gcca_dictionary["Trial_Type_Indexes"]


    # Plot Full Traces
    mean_visual_trace_list = []
    mean_odour_trace_list = []
    mean_transition_trace_list = []

    for session_index in range(number_of_sessions):

        # Get Full Tensor
        full_tensor = full_tensor_list[session_index]

        # Get Tensor Structure
        number_of_trials = np.shape(full_tensor)[0]
        trial_length = np.shape(full_tensor)[1]
        number_of_neurons = np.shape(full_tensor)[2]
        flat_tensor = np.reshape(full_tensor, (number_of_trials * trial_length, number_of_neurons))
        print("Flat Tensor Shape", np.shape(flat_tensor))
        print("Trial Length", trial_length)

        # Transform Data
        transformed_tensor = np.dot(flat_tensor, projection_matrixies[session_index])
        print("Transformed Tensor Shape", np.shape(transformed_tensor))

        # Get Trial Type Indexes
        trial_type_indexes = trial_type_index_list[session_index]
        visual_indicies = trial_type_indexes[0]
        odour_indicies = trial_type_indexes[1]
        perfect_transition_indicies = trial_type_indexes[2]
        imperfect_transition_indicies = trial_type_indexes[3]

        # Fill In Trial Type
        visual_traces = []
        for index in visual_indicies:
            start = index * trial_length
            stop = start + trial_length
            trace = transformed_tensor[start:stop, 2]
            visual_traces.append(trace)

        odour_traces = []
        for index in odour_indicies:
            start = index * trial_length
            stop = start + trial_length
            trace = transformed_tensor[start:stop, 2]
            odour_traces.append(trace)

        perfect_transition_traces = []
        for index in perfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            trace = transformed_tensor[start:stop, 2]
            perfect_transition_traces.append(trace)

        mean_visual_trace = np.mean(visual_traces, axis=0)
        mean_odour_trace = np.mean(odour_traces, axis=0)
        mean_transition_trace = np.mean(perfect_transition_traces, axis=0)

        mean_visual_trace_list.append(mean_visual_trace)
        mean_odour_trace_list.append(mean_odour_trace)
        mean_transition_trace_list.append(mean_transition_trace)

        # Plot Traces
    figure_1 = plt.figure()

    rows = 1
    columns = 3
    axis_1 = figure_1.add_subplot(rows, columns, 1)
    axis_2 = figure_1.add_subplot(rows, columns, 2)
    axis_3 = figure_1.add_subplot(rows, columns, 3)

    for trace in mean_visual_trace_list:
        print("trace")
        print(trace)
        axis_1.plot(trace)
        axis_1.set_ylim(-0.15, 0.15)

    for trace in mean_odour_trace_list:
        axis_2.plot(trace)
        axis_2.set_ylim(-0.15, 0.15)

    for trace in mean_transition_trace_list:
        axis_3.plot(trace)
        axis_3.set_ylim(-0.15, 0.15)

    plt.show()


def get_waveform_subsequent_trials(gcca_dictionary):
    number_of_sessions = len(gcca_dictionary["Session_Names"])
    full_tensor_list = gcca_dictionary["Full_Tensor_List"]
    projection_matrixies = gcca_dictionary["Projection_Matricies"]
    trial_type_index_list = gcca_dictionary["Trial_Type_Indexes"]

    # Plot Full Traces
    mean_transition_trace_list = []

    for session_index in range(number_of_sessions):

        # Get Full Tensor
        full_tensor = full_tensor_list[session_index]

        # Get Tensor Structure
        number_of_trials = np.shape(full_tensor)[0]
        trial_length = np.shape(full_tensor)[1]
        number_of_neurons = np.shape(full_tensor)[2]
        flat_tensor = np.reshape(full_tensor, (number_of_trials * trial_length, number_of_neurons))
        print("Flat Tensor Shape", np.shape(flat_tensor))
        print("Trial Length", trial_length)

        # Transform Data
        transformed_tensor = np.dot(flat_tensor, projection_matrixies[session_index])
        print("Transformed Tensor Shape", np.shape(transformed_tensor))

        # Get Trial Type Indexes
        trial_type_indexes = trial_type_index_list[session_index]
        perfect_transition_indicies = trial_type_indexes[2]

        perfect_transition_traces = []
        for index in perfect_transition_indicies:
            start = index * trial_length
            stop = start + 4 * trial_length
            trace = transformed_tensor[start:stop, 2]
            perfect_transition_traces.append(trace)

        mean_transition_trace = np.mean(perfect_transition_traces, axis=0)
        mean_transition_trace_list.append(mean_transition_trace)

        # Plot Traces
    figure_1 = plt.figure()

    rows = 1
    columns = 3
    axis_1 = figure_1.add_subplot(rows, columns, 1)

    for trace in mean_transition_trace_list:
        axis_1.plot(trace)

    plt.show()



def plot_trajectories_jpca(gcca_dictionary):

    number_of_sessions = len(gcca_dictionary["Session_Names"])
    full_tensor_list = gcca_dictionary["Full_Tensor_List"]
    projection_matrixies = gcca_dictionary["Projection_Matricies"]
    trial_type_index_list = gcca_dictionary["Trial_Type_Indexes"]

    # Plot Traces
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    # Plot Full Traces
    low_d_trajectory_tensor = []

    for session_index in range(number_of_sessions):

        # Get Full Tensor
        full_tensor = full_tensor_list[session_index]

        # Get Tensor Structure
        number_of_trials = np.shape(full_tensor)[0]
        trial_length = np.shape(full_tensor)[1]
        number_of_neurons = np.shape(full_tensor)[2]
        flat_tensor = np.reshape(full_tensor, (number_of_trials * trial_length, number_of_neurons))
        print("Flat Tensor Shape", np.shape(flat_tensor))
        print("Trial Length", trial_length)

        # Transform Data
        transformed_tensor = np.dot(flat_tensor, projection_matrixies[session_index])
        print("Transformed Tensor Shape", np.shape(transformed_tensor))

        # Get Trial Type Indexes
        trial_type_indexes = trial_type_index_list[session_index]
        visual_indicies = trial_type_indexes[0]
        odour_indicies = trial_type_indexes[1]
        perfect_transition_indicies = trial_type_indexes[2]
        imperfect_transition_indicies = trial_type_indexes[3]

        # Fill In Trial Type
        for index in visual_indicies:
            start = index * trial_length
            stop = start + trial_length
            # axis_1.plot(transformed_tensor[start:stop, 0], transformed_tensor[start:stop, 1], transformed_tensor[start:stop, 2], c='b', alpha=0.5)

        for index in odour_indicies:
            start = index * trial_length
            stop = start + trial_length
            # axis_1.plot(transformed_tensor[start:stop, 0], transformed_tensor[start:stop, 1], transformed_tensor[start:stop, 2], c='g',  alpha=0.5)

        for index in perfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.plot(transformed_tensor[start:stop, 0], transformed_tensor[start:stop, 1], transformed_tensor[start:stop, 2], c='m', alpha=0.5)

        for index in imperfect_transition_indicies:
            start = index * trial_length
            stop = start + trial_length
            axis_1.plot(transformed_tensor[start:stop, 0], transformed_tensor[start:stop, 1], transformed_tensor[start:stop, 2], c='orange', alpha=0.5)

    plt.show()



def compare_perfect_v_imperfect_transitions(gcca_dictionary):

    number_of_sessions = len(gcca_dictionary["Session_Names"])
    full_tensor_list = gcca_dictionary["Full_Tensor_List"]
    projection_matrixies = gcca_dictionary["Projection_Matricies"]
    trial_type_index_list = gcca_dictionary["Trial_Type_Indexes"]



    # Plot Traces
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    # Plot Full Traces
    low_d_trajectory_tensor = []

    for session_index in range(number_of_sessions):

        session_tensor = full_tensor_list[session_index]

        projection_matrix = projection_matrixies[session_index]
        transition_axis = projection_matrix[:,2]

        session_trial_indexes = trial_type_index_list[session_index]
        perfect_transition_indicies = session_trial_indexes[2]
        imperfect_transition_indicies = session_trial_indexes[3]

        print(np.shape(session_tensor))

        for trial_index in perfect_transition_indicies:
            trial_data = session_tensor[trial_index]
            transformed_trial_data = np.dot(trial_data, transition_axis)
            plt.plot(transformed_trial_data, c='b')

        for trial_index in imperfect_transition_indicies:
            trial_data = session_tensor[trial_index]
            transformed_trial_data = np.dot(trial_data, transition_axis)
            plt.plot(transformed_trial_data, c='r')

        plt.show()


"""
for component_index in range(number_of_components):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.set_title("Component" + str(component_index))
    for session_index in range(number_of_sessions):
        axis_1.plot(transormed_data[session_index, :, component_index])

        component_max = np.max(transormed_data[:, :, component_index])
        for x in range(0, trial_length*number_of_conditions, trial_length):
            axis_1.vlines(x=x, ymin=0, ymax=component_max)


    plt.show()
"""


# Get Full Session Traces
"""
all_flat_tensors = []
for session_index in range(number_of_sessions):

    # Get Full Tensor
    full_tensor = full_tensor_list[session_index]

    # Get Tensor Structure
    number_of_trials = np.shape(full_tensor)[0]
    trial_length = np.shape(full_tensor)[1]
    number_of_neurons = np.shape(full_tensor)[2]

    # Reshape Tensor
    reshaped_tensor = np.ndarray.reshape(full_tensor, (number_of_trials * trial_length, number_of_neurons))

    # Add To List
    all_flat_tensors.append(reshaped_tensor)

# Transform Full Data
transformed_full_data_list = gcca.transform(all_flat_tensors)
"""

gcca_dictionary_location = "/home/matthew/Documents/GCCA_Analysis/GCCA_Output_Dictionary.npy"
gcca_dictionary = np.load(gcca_dictionary_location, allow_pickle=True)[()]


#view_trial_averaged_components(gcca_dictionary)
#plot_factor_decay_perfect_v_imperfect(gcca_dictionary)
#draw_trial(gcca_dictionary)
#plot_gcca_trajectories(gcca_dictionary)
#get_mean_waveforms(gcca_dictionary)
#get_waveform_subsequent_trials(gcca_dictionary)
#plot_factor_decay(gcca_dictionary)
#view_projection_matricies(gcca_dictionary)
##print(gcca_dictionary)

compare_perfect_v_imperfect_transitions(gcca_dictionary)