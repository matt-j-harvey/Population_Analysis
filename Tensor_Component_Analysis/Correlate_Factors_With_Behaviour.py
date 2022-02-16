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

def list_folders(root_directory):
    folder_list = []
    all_files = os.listdir(root_directory)
    for file_candidate in all_files:
        if os.path.isdir(os.path.join(root_directory, file_candidate)):
            folder_list.append(os.path.join(root_directory, file_candidate))
    return folder_list



def normalise_trial_loadings(trial_loadings):

    # Subtract Min, so Min is zero
    min_vector = np.min(trial_loadings, axis=0)
    trial_loadings = np.subtract(trial_loadings, min_vector)

    # Divide By Max, so Max is one
    max_vector = np.max(trial_loadings, axis=0)
    trial_loadings = np.divide(trial_loadings, max_vector)

    return trial_loadings



def get_factor_switch_modulation(base_directory, save_directory):


    # Load Trial Components
    trial_loadings = np.load(os.path.join(base_directory, "trial_loadings.npy"))
    visual_blocks = np.load(os.path.join(base_directory,  "visual_blocks.npy"))
    odour_blocks = np.load(os.path.join(base_directory,   "odour_blocks.npy"))

    switch_indexes = np.load(os.path.join(base_directory, "switch_indicies.npy"))
    perfect_switch_trials = np.load(os.path.join(base_directory, "perfect_switch_trials.npy"))

    print("Switch Indicies", switch_indexes)
    print("Perfect switch trials", perfect_switch_trials)

    # Normalise Trial Loadings:
    trial_loadings = normalise_trial_loadings(trial_loadings)


    # Get All Switch Indicies
    all_switch_indicies = []
    for block in odour_blocks:
        print(block)
        all_switch_indicies.append(block[1] + 1)
    for block in visual_blocks:
        print("Visual Block", block)

    number_of_factors = np.shape(trial_loadings)[1]

    figure_1 = plt.figure(constrained_layout=True)
    figure_1_grid_spec = gridspec.GridSpec(ncols=4, nrows=number_of_factors, figure=figure_1)
    trial_axis_list = []
    difference_axis_list = []

    for factor in range(number_of_factors):

        # Create Axis
        trial_axis = figure_1.add_subplot(figure_1_grid_spec[factor, 0:2])
        difference_axis = figure_1.add_subplot(figure_1_grid_spec[factor, 2])
        perfection_axis = figure_1.add_subplot(figure_1_grid_spec[factor, 3])

        # Plot Trial Data
        factor_data = trial_loadings[:, factor]
        trial_axis.plot(factor_data)
        trial_axis.vlines(all_switch_indicies, ymin=np.min(factor_data), ymax=np.max(factor_data), color='k')

        number_of_selected_switches = len(switch_indexes)
        for selected_switch_index in range(number_of_selected_switches):
            switch_index = switch_indexes[selected_switch_index]
            outcome = perfect_switch_trials[selected_switch_index]
            if outcome == 1:
                colour = 'tab:purple'
            else:
                colour = 'darkorange'
            trial_axis.scatter([switch_index], [1], c=colour)

        # Highlight Blocks
        for block in visual_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='blue')
        for block in odour_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='green')

        # Get Pre and Post Loadings
        colourmap = cm.get_cmap('tab10')

        number_of_switches = len(all_switch_indicies)
        for switch_index in range(number_of_switches):
            switch = all_switch_indicies[switch_index]
            pre_activity = factor_data[switch - 1]
            #post_activity = np.mean(factor_data[switch])
            post_activity = np.mean(factor_data[switch: switch + 3])

            colour = colourmap(float(switch_index)/number_of_switches)
            print(colour)
            difference_axis.plot([0, 1], [pre_activity, post_activity], c=colour)
            difference_axis.scatter([0], [pre_activity], c=colour)
            difference_axis.scatter([1], [post_activity], c=colour)

        difference_axis.set_ylim([0, 1])


        # Get Perfect V Imperfect Switch Values
        perfect_changes = []
        imperfect_changes = []
        for switch_index in range(len(switch_indexes)):
                switch = switch_indexes[switch_index]
                change = factor_data[switch] - factor_data[switch-1]
                if perfect_switch_trials[switch_index] == 1:
                    perfect_changes.append(change)
                elif perfect_switch_trials[switch_index] == 0:
                    imperfect_changes.append(change)

        for value in imperfect_changes:
            perfection_axis.scatter([0], [value], c='b')
        for value in perfect_changes:
            perfection_axis.scatter([1], [value], c='g')


        trial_axis_list.append(trial_axis)
        difference_axis_list.append(difference_axis)

    session_name = base_directory.split("/")[-1]
    plt.savefig(save_directory + "/" + session_name + ".png")
    plt.close()


root_directory = "/media/matthew/29D46574463D2856/Nick_TCA_Plots/"
directory_list = list_folders(root_directory)

save_directory = r"/home/matthew/Pictures/TCA_Factor_Switch_Correlation/"
for directory in directory_list:
    get_factor_switch_modulation(directory, save_directory)