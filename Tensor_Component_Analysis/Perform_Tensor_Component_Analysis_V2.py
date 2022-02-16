import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import AffinityPropagation
from mpl_toolkits import mplot3d
from matplotlib.pyplot import cm
from tensorly.decomposition import parafac, CP, non_negative_parafac
import os

import Import_Preprocessed_Data
import Load_Data_For_TCA
import Create_Behaviour_Matrix_Downsampled_AI




def plot_factors(trial_loadings, time_loadings, stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies, trial_start, session_name, save_directory):

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

        # Scatter Trial Indexes
        trial_max = np.max(trial_data)
        trial_axis.scatter(stable_visual_indicies,          np.multiply(np.ones(len(stable_visual_indicies)),           trial_max), c='b')
        trial_axis.scatter(stable_odour_indicies,           np.multiply(np.ones(len(stable_odour_indicies)),            trial_max), c='g')
        trial_axis.scatter(imperfect_transition_indicies,   np.multiply(np.ones(len(imperfect_transition_indicies)),    trial_max), c='r')
        trial_axis.scatter(perfect_transition_indicies,     np.multiply(np.ones(len(perfect_transition_indicies)),      trial_max), c='m')

        # Mark Stimuli Onset
        time_axis.vlines(0-trial_start, ymin=np.min(time_data), ymax=np.max(time_data), color='k')


    figure_1.set_size_inches(18.5, 16)
    #figure_1.tight_layout()
    plt.savefig(save_directory + "/" + session_name + ".png", dpi=200)
    plt.close()



def perform_tensor_component_analysis(file_list, base_directory, behaviour_matrix_directory, plot_save_directory, trial_start=-6, trial_stop=17, number_of_factors=7, onset_or_offset='offset'):

    for matlab_file_location in file_list:

        # Create Behaviour Matrix
        #Create_Behaviour_Matrix_Downsampled_AI.create_behaviour_matrix(matlab_file_location, behaviour_matrix_directory)

        # Get Session Name
        session_name = matlab_file_location.split('/')[-1]
        session_name = session_name.replace("_preprocessed_basic.mat", "")
        print("Performing Tensor Component Analysis for Session: ", session_name)

        # Load Data
        trial_tensor, stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies = Load_Data_For_TCA.load_data_for_tca(matlab_file_location, behaviour_matrix_directory, trial_start, trial_stop, onset_or_offset)

        # Perform Tensor Decomposition
        weights, factors = non_negative_parafac(trial_tensor, rank=number_of_factors, init='svd', verbose=1, n_iter_max=500)
        #weights, factors = parafac(trial_tensor, rank=number_of_factors, init='random', verbose=1, n_iter_max=500)

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

        np.save(os.path.join(factor_save_directory, "stable_visual_indicies.npy"),  stable_visual_indicies)
        np.save(os.path.join(factor_save_directory, "stable_odour_indicies.npy"), stable_odour_indicies)
        np.save(os.path.join(factor_save_directory, "perfect_transition_indicies.npy"), perfect_transition_indicies)
        np.save(os.path.join(factor_save_directory, "imperfect_transition_indicies.npy"), imperfect_transition_indicies)

        plot_factors(trial_loadings, time_loadings, stable_visual_indicies, stable_odour_indicies, perfect_transition_indicies, imperfect_transition_indicies, trial_start, session_name, plot_save_directory)


def load_matlab_sessions(base_directory):

    matlab_file_list = []
    all_files = os.listdir(base_directory)
    for file in all_files:
        if file[-3:] == "mat":
            matlab_file_list.append(os.path.join(base_directory, file))

    return matlab_file_list


# Set Number of Factors
number_of_factors = 30
trial_start = -6
trial_stop = 26 #Should be 18 if offset

# Load Matlab Data
base_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/Best_switching_sessions_all_sites"
plot_save_directory = "/home/matthew/Pictures/TCA_Plots/Best_All_Sites/"
behaviour_matrix_directory = r"/home/matthew/Documents/Nick_Population_Analysis_Data/python_export/Behaviour_Matricies"

file_list = load_matlab_sessions(base_directory)
perform_tensor_component_analysis(file_list, base_directory, behaviour_matrix_directory, plot_save_directory,  trial_start=trial_start, trial_stop=trial_stop, number_of_factors=number_of_factors, onset_or_offset='onset')

