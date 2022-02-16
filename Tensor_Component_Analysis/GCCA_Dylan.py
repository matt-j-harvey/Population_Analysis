import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
import os
from scipy.io import loadmat
import mat73

from mvlearn.embed import GCCA



def plot_gcca_mean_traces(transformed_data, number_of_sessions, number_of_components, unique_condition_labels, condition_length, number_of_conditions, number_of_timepoints):

    cmap = cm.get_cmap('gist_rainbow')
    

    for component_index in range(number_of_components):

        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1,1,1)
        colour_list = []

        for session_index in range(number_of_sessions):

            component_trace = transformed_data[session_index][:, component_index]
            axis_1.plot(component_trace)

        # Draw Black Lines To Demarcate Conditions
        for x in range(0, number_of_timepoints, condition_length):
            axis_1.axvline(x, ymin=-1, ymax=1, color='k')

        # Shade Conditions
        for condition_index in range(0, number_of_conditions):
            condition_colour = cmap(float(condition_index) / number_of_conditions)
            colour_list.append(condition_colour)

            start = condition_index * condition_length
            stop = start + condition_length

            axis_1.axvspan(xmin=start, xmax=stop-1, facecolor=condition_colour, alpha=0.3)

        # Add Legengs
        patch_list = []
        for condition_index in range(number_of_conditions):
            patch = mpatches.Patch(color=colour_list[condition_index], label=unique_condition_labels[condition_index], alpha=0.3)
            patch_list.append(patch)

        axis_1.legend(handles=patch_list)
        axis_1.set_title("Component: " + str(component_index))
        plt.show()



def reshape_neural_data(neural_data_list):

    transformed_data = []

    for raster in neural_data_list:
        dropped_raster = raster[0]
        transposed_raster = np.transpose(dropped_raster)
        transposed_raster = np.nan_to_num(transposed_raster)
        transformed_data.append(transposed_raster)

    return transformed_data


def filter_by_opsin(neural_data_list, opsin_labels, selected_label):

    filtered_data = []
    number_of_sessions = len(neural_data_list)

    for session_index in range(number_of_sessions):

        opsin_type = opsin_labels[session_index][0]


        if opsin_type == selected_label:
            filtered_data.append(neural_data_list[session_index])

    return filtered_data



def load_condition_labels(matlab_data):


    condition_labels = matlab_data['cond_labels'][0]
    condition_labels_set = set(condition_labels)
    number_of_conditions = len(list(condition_labels_set))
    number_of_timepoints = len(condition_labels)
    condition_length = int(number_of_timepoints / number_of_conditions)

    print("Number of timepoints: ", number_of_timepoints)
    print("Number of conditions: ", number_of_conditions)
    print("Condition Length: ", condition_length)

    unique_condition_labels = []
    for x in range(0, number_of_timepoints-1, condition_length):
        unique_condition_labels.append(condition_labels[x])

    return unique_condition_labels, condition_length, number_of_conditions, number_of_timepoints




# Set File Location
file_location = r"//media/matthew/29D46574463D2856/Dylan_Population_Data/Dylan_all_VIP_mice_GCCA_data.mat"

# Load Matlab Data
matlab_data = mat73.loadmat(file_location)
matlab_data = matlab_data['output_GCCA']

# Get Opsin Labels
opsin_labels = matlab_data['opsin']
unique_condition_labels, condition_length, number_of_conditions, number_of_timepoints = load_condition_labels(matlab_data)

# Load and Transform Neural Data
neural_data = matlab_data['neural_data']
neural_data = reshape_neural_data(neural_data)

# Filter By Opsin If You Want:
neural_data = filter_by_opsin(neural_data, opsin_labels, selected_label='tdt')


# Perform GCCA
number_of_sessions = len(neural_data)
number_of_components = 30
gcca_model =GCCA(n_components=number_of_components)
transformed_data = gcca_model.fit_transform(neural_data)

# Plot GCCA Data
plot_gcca_mean_traces(transformed_data, number_of_sessions, number_of_components, unique_condition_labels, condition_length, number_of_conditions, number_of_timepoints)