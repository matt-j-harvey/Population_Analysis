import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm

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


def perform_dimensionality_reduction(trial_tensor, n_components=3):

    # Get Tensor Shape
    number_of_trials = np.shape(trial_tensor)[0]
    trial_length = np.shape(trial_tensor)[1]
    number_of_neurons = np.shape(trial_tensor)[2]

    # Flatten Tensor To Perform Dimensionality Reduction
    reshaped_tensor = np.reshape(trial_tensor, (number_of_trials * trial_length, number_of_neurons))

    # Perform Dimensionality Reduction
    model = FactorAnalysis(n_components=n_components)
    model.fit(reshaped_tensor)

    transformed_data = model.transform(reshaped_tensor)
    components = model.components_

    # Put Transformed Data Back Into Tensor Shape
    transformed_data = np.reshape(transformed_data, (number_of_trials, trial_length, n_components))

    return components, transformed_data


def plot_trajectories(transformed_data, labels, trial_order, switch_indicies, save_directory, session_name, colouring="Context"):

    figure_1 = plt.figure()
    ax = figure_1.gca(projection='3d')
    number_of_trials = np.shape(transformed_data)[0]
    colour_map = cm.get_cmap('gist_rainbow')

    for trial_index in range(number_of_trials):

        if colouring == "Context":
            colour_value = labels[trial_index]
            colour = colour_map(colour_value)
            ax.plot(transformed_data[trial_index, :, 0], transformed_data[trial_index, :, 1],
                    transformed_data[trial_index, :, 2], c=colour, alpha=0.5)

        elif colouring == "Switch":
            if trial_index in switch_indicies:
                colour_value = 0.5
                alpha = 1
            else:
                colour_value = labels[trial_index]
                alpha = 0.1
            colour = colour_map(colour_value)
            ax.plot(transformed_data[trial_index, :, 0], transformed_data[trial_index, :, 1],
                    transformed_data[trial_index, :, 2], c=colour, alpha=alpha)

        elif colouring == "Time":
            colour_map = cm.get_cmap('plasma')
            colour_value = trial_order[trial_index]
            colour = colour_map(colour_value)
            ax.plot(transformed_data[trial_index, :, 0], transformed_data[trial_index, :, 1],
                    transformed_data[trial_index, :, 2], c=colour, alpha=0.5)

        elif colouring == "Trial":
            trial_length = np.shape(transformed_data)[1]
            colour_values = list(range(trial_length))
            colour_values = np.divide(colour_values, trial_length)
            colour_map = cm.get_cmap('plasma')
            for point in range(trial_length-1):
                x_start = transformed_data[trial_index, point,     0]
                x_stop  = transformed_data[trial_index, point + 1, 0]
                y_start = transformed_data[trial_index, point,     1]
                y_stop  = transformed_data[trial_index, point + 1, 1]
                z_start = transformed_data[trial_index, point,     2]
                z_stop  = transformed_data[trial_index, point + 1, 2]
                colour_value = colour_values[point]
                colour = colour_map(colour_value)
                ax.plot([x_start, x_stop], [y_start, y_stop], [z_start, z_stop], color=colour, alpha=0.5)

    # Save File Name
    save_file_name = save_directory + "/" + session_name + "_" + colouring + ".avi"
    create_3d_video(figure_1, ax, save_file_name)
    #plt.savefig(save_directory + "/" + session_name + "_" + colouring + ".png")
    #plt.close()

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
    delta_f_matrix = np.nan_to_num(delta_f_matrix)
    return delta_f_matrix


def create_3d_video(figure, axis, video_name):
    width = None
    height = None
    fps = 15

    # Get Video Height and Width
    width, height = figure.canvas.get_width_height()

    # Create Video
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), frameSize=(width, height), fps=15)  # 0, 12

    # Rotate Axis 360 Degrees
    for angle in range(360):
       
        # Set Rotation
        axis.view_init(30, angle)
        figure.canvas.draw()
        
        # Grab Figure Image 
        img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8,sep='')
        img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
     
        # Write To Video File
        video.write(img)
        
    cv2.destroyAllWindows()
    video.release()



def view_neural_trajectories(file_list, save_directory, trial_start=-10, trial_stop=42):


    for matlab_file_location in file_list:

        # Get Session Name
        session_name = matlab_file_location.split('/')[-1]
        print("Getting Neural Trajectories for Session: ", session_name)

        # Load Matalb Data
        data_object = Import_Preprocessed_Data.ImportMatLabData(matlab_file_location)

        # Create Subdirectory For This Session
        sub_save_directory = save_directory + "/" + session_name
        if not os.path.exists(sub_save_directory):
            os.mkdir(sub_save_directory)

        # Extract Delta F Matrix
        delta_f_matrix = data_object.dF
        delta_f_matrix = smooth_delta_f_matrix(delta_f_matrix)
        delta_f_matrix = normalise_delta_f_matrix(delta_f_matrix)
        print("Delta F Matrix Shape", np.shape(delta_f_matrix))

        # Extract Onets Of Interest
        visual_context_vis_2_frames = data_object.vis1_frames[0]
        odour_context_vis_2_frames = data_object.irrel_vis1_frames[0]
        expected_odour_trials = data_object.mismatch_trials['exp_odour'][0]
        all_vis_2_onsets = np.concatenate([visual_context_vis_2_frames, odour_context_vis_2_frames])

        # Create Trial Tensor
        vis_2_trial_tensor = create_trial_tensor(delta_f_matrix, all_vis_2_onsets, trial_start, trial_stop)

        # Perform Dimensionality Reduction
        components, transformed_data = perform_dimensionality_reduction(vis_2_trial_tensor)

        # Create Apropriate Colour Labels
        labels = []
        for onset in visual_context_vis_2_frames:
            labels.append(0.2)

        for onset in odour_context_vis_2_frames:
            labels.append(0.8)

        #  Get Trial Orderings
        trial_orders = []
        ordered_onset_list = np.copy(all_vis_2_onsets)
        ordered_onset_list.sort()
        ordered_onset_list = list(ordered_onset_list)
        for onset in all_vis_2_onsets:
            trial_index = ordered_onset_list.index(onset)
            trial_orders.append(trial_index)

        # Get Switch Indicies
        switch_indicies = []
        for trial in expected_odour_trials:
            switch_indicies.append(list(all_vis_2_onsets).index(trial))

        # Plot Trajectories
        plot_trajectories(transformed_data, labels, trial_orders, switch_indicies, sub_save_directory, session_name, colouring="Context")
        plot_trajectories(transformed_data, labels, trial_orders, switch_indicies, sub_save_directory, session_name, colouring="Time")
        plot_trajectories(transformed_data, labels, trial_orders, switch_indicies, sub_save_directory, session_name, colouring="Trial")
        plot_trajectories(transformed_data, labels, trial_orders, switch_indicies, sub_save_directory, session_name, colouring="Switch")



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


save_directory = "/home/matthew/Pictures/Nick_Factor_Analysis_Plots/"
view_neural_trajectories(file_list, save_directory)