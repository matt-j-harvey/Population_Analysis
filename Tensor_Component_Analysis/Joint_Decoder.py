import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import FactorAnalysis, TruncatedSVD, FastICA, PCA, NMF
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
import random


def create_correlation_tensor(activity_matrix, onsets, start_window, stop_window):

    # Get Tensor Details
    number_of_clusters = np.shape(activity_matrix)[0]
    number_of_trials = np.shape(onsets)[0]

    # Create Empty Tensor To Hold Data
    correlation_tensor = np.zeros((number_of_trials, number_of_clusters, number_of_clusters))

    # Get Correlation Matrix For Each Trial
    for trial_index in range(0, number_of_trials):

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[:, trial_start:trial_stop]

        print("Trial: ", trial_index, " of ", number_of_trials, " Onset: ", trial_start, " Offset: ", trial_stop)

        # Get Trial Correlation Matrix
        trial_correlation_matrix = np.zeros((number_of_clusters, number_of_clusters))

        for cluster_1_index in range(number_of_clusters):
            cluster_1_trial_trace = trial_activity[cluster_1_index]

            for cluster_2_index in range(cluster_1_index + 1, number_of_clusters):
                cluster_2_trial_trace = trial_activity[cluster_2_index]

                correlation = np.corrcoef(cluster_1_trial_trace, cluster_2_trial_trace)[0][1]

                trial_correlation_matrix[cluster_1_index][cluster_2_index] = correlation
                trial_correlation_matrix[cluster_2_index][cluster_1_index] = correlation

        correlation_tensor[trial_index] = trial_correlation_matrix

    return correlation_tensor



def get_block_boundaries(combined_onsets, visual_context_onsets, odour_context_onsets):

    visual_blocks = []
    odour_blocks = []

    current_block_start = 0
    current_block_end = None

    # Get Initial Onset
    if combined_onsets[0] in visual_context_onsets:
        current_block_type = 0
    elif combined_onsets[0] in odour_context_onsets:
        current_block_type = 1
    else:
        print("Error! onsets not in either vidual or oflactory onsets")

    # Iterate Through All Subsequent Onsets
    number_of_onsets = len(combined_onsets)
    for onset_index in range(1, number_of_onsets):

        # Get Onset
        onset = combined_onsets[onset_index]

        # If we are currently in an Visual Block
        if current_block_type == 0:

            # If The Next Onset is An Odour Block - Block Finish, add Block To Boundaries
            if onset in odour_context_onsets:
                current_block_end = onset_index-1
                visual_blocks.append([current_block_start, current_block_end])
                current_block_type = 1
                current_block_start = onset_index

        # If we Are currently in an Odour BLock
        if current_block_type == 1:

            # If The NExt Onset Is a Visual Trial - BLock Finish Add Block To Block Boundaires
            if onset in visual_context_onsets:
                current_block_end = onset_index - 1
                odour_blocks.append([current_block_start, current_block_end])
                current_block_type = 0
                current_block_start = onset_index

    return visual_blocks, odour_blocks


def perform_factor_analysis(tensor, n_components=7):

    # Remove Nans
    tensor = np.nan_to_num(tensor)

    # Get Tensor Structure
    number_of_trials = np.shape(tensor)[0]
    number_of_clusters = np.shape(tensor)[1]

    # Concatenate Trials
    tensor = np.reshape(tensor, (number_of_trials, number_of_clusters * number_of_clusters))

    print("Reshaped Tensor Shape", np.shape(tensor))

    # Perform Factor Analysis
    #tensor = np.clip(tensor, a_min=0, a_max=None)
    model = FactorAnalysis(n_components=n_components)
    model.fit(tensor)

    # Get Components
    components = model.components_

    # Factor Trajectories
    low_dimensional_trajectories = model.transform(tensor)

    print("Trajectories Shape", np.shape(low_dimensional_trajectories))

    return components,  low_dimensional_trajectories




def plot_factors_combined(trial_loadings, weight_loadings, visual_blocks, odour_blocks):

    print("Weight Data Shape", np.shape(weight_loadings))
    print("TRial Loadings Shape", np.shape(trial_loadings))

    number_of_factors = np.shape(trial_loadings)[1]
    number_of_correlations = np.shape(weight_loadings)[1]
    number_of_clusters = int(math.sqrt(number_of_correlations))

    print("Number of Factors", number_of_factors)
    print("Number of correlations", number_of_correlations)
    print("Number of clusters", number_of_clusters)

    rows = number_of_factors
    columns = 2

    figure_count = 1
    figure_1 = plt.figure()
    #figure_1.suptitle(session_name)
    for factor in range(number_of_factors):
        weights_axis = figure_1.add_subplot(rows,  columns, figure_count)
        trial_axis = figure_1.add_subplot(rows, columns, figure_count + 1)
        figure_count += 2

        weights_axis.set_title("Factor " + str(factor) + " Weight Loadings")
        trial_axis.set_title("Factor " + str(factor) + " Trial Loadings")

        weight_data = weight_loadings[factor]
        trial_data = trial_loadings[:, factor]

        # Plot Weight Matrix
        weight_data = np.reshape(weight_data, (number_of_clusters, number_of_clusters))
        weights_axis.imshow(weight_data, cmap='bwr')

        trial_axis.plot(trial_data, c='orange')

        # Highligh Blocks
        for block in visual_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='blue')
        for block in odour_blocks:
            trial_axis.axvspan(block[0], block[1], alpha=0.2, color='green')

    figure_1.set_size_inches(18.5, 16)
    figure_1.tight_layout()
    plt.show()
    plt.close()



def get_correlation_tensors(session_list):

    # Trial Settings
    trial_start = -70
    trial_stop = -14

    for base_directory in session_list:

        # Load Activity Matrix
        cluster_activity_matrix_file = base_directory + "/Cluster_Activity_Matrix.npy"
        activity_matrix = np.load(cluster_activity_matrix_file)
        activity_matrix = np.nan_to_num(activity_matrix)

        # Load Stimuli Onsets
        stimuli_onsets_directory = base_directory + "/Stimuli_Onsets/"
        visual_context_onsets_vis_1 = np.load(stimuli_onsets_directory + "visual_context_stable_vis_1_frame_onsets.npy")
        visual_context_onsets_vis_2 = np.load(stimuli_onsets_directory + "visual_context_stable_vis_2_frame_onsets.npy")
        odour_context_onsets_vis_1 = np.load(stimuli_onsets_directory + "odour_context_stable_vis_1_frame_onsets.npy")
        odour_context_onsets_vis_2 = np.load(stimuli_onsets_directory + "odour_context_stable_vis_2_frame_onsets.npy")

        # Arrange Onsets
        visual_context_onsets = np.concatenate([visual_context_onsets_vis_1, visual_context_onsets_vis_2])
        odour_context_onsets = np.concatenate([odour_context_onsets_vis_1, odour_context_onsets_vis_2])

        # Create Correlation Tensors
        visual_correlation_tensor = create_correlation_tensor(activity_matrix, visual_context_onsets, trial_start, trial_stop)
        odour_correlation_tensor = create_correlation_tensor(activity_matrix, odour_context_onsets, trial_start, trial_stop)

        # Save Trial Tensor
        save_directory = base_directory + "/Pre_Stimulus/"
        np.save(save_directory + "Visual_Context_Correlation_Tensor.npy", visual_correlation_tensor)
        np.save(save_directory + "Odour_Context_Correlation_Tensor.npy", odour_correlation_tensor)



def select_subset(tensor, sample_size):

    selected_datapoints = []
    number_of_samples = np.shape(tensor)[0]
    index_list = list(range(number_of_samples))
    random.shuffle(index_list)
    selected_indexes = index_list[0:sample_size]
    for index in selected_indexes:
        selected_datapoints.append(tensor[index])

    selected_datapoints = np.array(selected_datapoints)

    return selected_datapoints


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix






def view_weights():

    # Load WEights
    weights = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/joint_decoder_weights.npy")
    number_of_connections = np.shape(weights)[1]
    print("Weights Shape", np.shape(weights))

    number_of_clusters = int(math.sqrt(number_of_connections))
    print("Number of clusters", number_of_clusters)

    weights = np.reshape(weights, (number_of_clusters, number_of_clusters))

    base_directory = r"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/"
    draw_brain_network(base_directory, weights)


    # Split Into Positive and Negative Weights
    print("Positive Weights")
    positive_weights = np.where(weights > 0, weights, 0)
    draw_brain_network(base_directory, positive_weights, cmap='jet')

    print("Negative Weights")
    negative_weights = np.where(weights < 0, weights, 0)
    negative_weights = np.abs(negative_weights)
    draw_brain_network(base_directory, negative_weights, cmap='jet')


    # Decompose Regression Matrix
    cluster_regression_matrix(base_directory, weights)

    weights = sort_matrix(weights)
    magnitude = np.max(np.abs(weights))

    plt.imshow(weights, cmap='bwr', vmin=-1 * magnitude, vmax=magnitude)
    plt.show()


def cluster_regression_matrix(base_directory, weight_matrix):

    print("Weight Matrix Shaoe", np.shape(weight_matrix))
    number_of_regions = np.shape(weight_matrix)[0]

    number_of_clusters = 5
    model = SpectralClustering(n_clusters=number_of_clusters)
    model.fit(weight_matrix)

    # Get Labels
    labels = model.labels_
    print("labels", labels)

    for cluster in range(number_of_clusters):
        cluster_indicies = []

        for region in range(number_of_regions):
            region_label = labels[region]

            if region_label == cluster:
                cluster_indicies.append(region)


        cluster_correlation_matrix = np.zeros(np.shape(weight_matrix))
        for region in cluster_indicies:
            cluster_correlation_matrix[region] = weight_matrix[region]
            cluster_correlation_matrix[:, region] = weight_matrix[:, region]

        sorted_cluster_correlation_matrix = sort_matrix(cluster_correlation_matrix)
        plt.imshow(sorted_cluster_correlation_matrix)
        plt.show()
        draw_brain_network(base_directory, cluster_correlation_matrix)

    for component in components:
        print(np.shape(component))
        draw_brain_network(base_directory, component)


def draw_brain_network(base_directory, adjacency_matrix, cmap='bwr'):

    # Load Cluster Centroids
    cluster_centroids = np.load(base_directory + "/Cluster_Centroids.npy")

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(np.abs(weights)))

    # Get Edge Colours
    colourmap = cm.get_cmap(cmap)
    colours = []
    for weight in weights:
        colour = colourmap(weight)
        colours.append(colour)

    # Load Cluster Outlines
    cluster_outlines = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters_outline.npy")
    plt.imshow(cluster_outlines, cmap='binary')

    image_height = np.shape(cluster_outlines)[0]

    # Draw Graph
    # Invert Cluster Centroids
    inverted_centroids = []
    for centroid in cluster_centroids:
        y_value = centroid[1]
        x_value = centroid[0]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, inverted_y])

    #plt.title(session_name)
    nx.draw(graph, pos=inverted_centroids, node_size=1,  width=weights, edge_color=colours)
    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()
    plt.show()







def perform_decoding(session_list):

    # Load All Tensors
    visual_tensor_list = []
    odour_tensor_list = []
    tensor_sizes = []

    # Load Tensors
    for base_directory in session_list:
        save_directory = base_directory + "/Pre_Stimulus/"
        visual_tensor = np.load(save_directory + "Visual_Context_Correlation_Tensor.npy")
        odour_tensor = np.load(save_directory + "Odour_Context_Correlation_Tensor.npy")

        vis_trials = np.shape(visual_tensor)[0]
        odour_trials = np.shape(odour_tensor)[0]
        number_of_clusters = np.shape(visual_tensor)[1]

        visual_tensor = np.reshape(visual_tensor, (vis_trials, number_of_clusters * number_of_clusters))
        odour_tensor = np.reshape(odour_tensor, (odour_trials, number_of_clusters * number_of_clusters))

        tensor_sizes.append(vis_trials)
        tensor_sizes.append(odour_trials)

        visual_tensor_list.append(visual_tensor)
        odour_tensor_list.append(odour_tensor)


    # Split Into Test and Train Data
    sample_size = np.min(tensor_sizes)
    print("Sample Size", sample_size)
    balanced_visual_tensors = []
    balanced_odour_tensors = []

    for tensor in visual_tensor_list:
        balanced_tensor = select_subset(tensor, sample_size)
        balanced_visual_tensors.append(balanced_tensor)

    for tensor in odour_tensor_list:
        balanced_tensor = select_subset(tensor, sample_size)
        balanced_odour_tensors.append(balanced_tensor)


    # Combine Into Datasets
    datasets = []
    labels_list = []
    number_of_sessions = len(balanced_visual_tensors)

    for session in range(number_of_sessions):

        visual_data = balanced_visual_tensors[session]
        odour_data = balanced_odour_tensors[session]

        odour_labels = np.ones(sample_size)
        visual_labels = np.zeros(sample_size)

        data = np.vstack([visual_data, odour_data])
        labels = np.hstack([visual_labels, odour_labels])

        print("Datasets shape", np.shape(data))
        print("Labels Shape", np.shape(labels))

        datasets.append(data)
        labels_list.append(labels)


    # Split Into K Folds
    number_of_folds = 4
    skf = StratifiedKFold(n_splits=number_of_folds)
    train_index_list = []
    test_index_list = []

    for session in range(number_of_sessions):
        skf.get_n_splits(datasets[session], labels_list[session])
        train_fold_list = []
        test_fold_list = []

        print(np.shape(datasets[session]))
        print(np.shape(labels_list[session]))
        for train_index, test_index in skf.split(datasets[session], labels_list[session]):
            train_fold_list.append([train_index])
            test_fold_list.append([test_index])

        train_index_list.append(train_fold_list)
        test_index_list.append(test_fold_list)


    # Train and Test Model
    model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5, C=0.2)
    k_fold_session_scores = []
    model_weights_list = []
    for fold in range(number_of_folds):

        combined_train_data = []
        combined_train_labels = []

        combined_test_data = []
        combined_test_labels = []

        for session in range(number_of_sessions):
            train_indexes = train_index_list[session][fold]
            test_indexes = test_index_list[session][fold]

            session_train_data = datasets[session][train_indexes]
            session_train_labels = labels_list[session][train_indexes]

            for datapoint in session_train_data:
                combined_train_data.append(datapoint)

            for label in session_train_labels:
                combined_train_labels.append(label)

            session_test_data = datasets[session][test_indexes]
            session_test_labels = labels_list[session][test_indexes]

            combined_test_data.append(session_test_data)
            combined_test_labels.append(session_test_labels)

        # Train Model
        model.fit(combined_train_data, combined_train_labels)

        # Score Model
        session_scores = []
        for session in range(number_of_sessions):
            y_pred = model.predict(combined_test_data[session])
            y_true = combined_test_labels[session]
            accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
            session_scores.append(accuracy)

        k_fold_session_scores.append(session_scores)
        print("Fold: ", fold, "Session Scores", session_scores)

        # Get Coefficients
        coefficients = model.coef_
        model_weights_list.append(coefficients)

    k_fold_session_scores = np.array(k_fold_session_scores)
    mean_scores = np.mean(k_fold_session_scores, axis=0)
    print("Mean Scores", mean_scores)

    # Get Average Weights
    model_weights_list = np.array(model_weights_list)
    print("Weights shape", np.shape(model_weights_list))

    mean_model_weights = np.mean(model_weights_list, axis=0)

    np.save("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/joint_decoder_weights.npy", mean_model_weights)





session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/"]

#get_correlation_tensors(session_list)
perform_decoding(session_list)
view_weights()