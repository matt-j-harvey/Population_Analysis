import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import FactorAnalysis, TruncatedSVD, FastICA, PCA, NMF
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, DBSCAN, AffinityPropagation
from sklearn.mixture import GaussianMixture
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
import random


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix



def draw_brain_network(base_directory, adjacency_matrix, axis, cmap='bwr'):

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
    axis.imshow(cluster_outlines, cmap='binary')

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
    nx.draw(graph, pos=inverted_centroids, node_size=1,  width=weights, edge_color=colours, ax=axis)


def get_correlational_changes(session_list):

    # Load All Tensors
    visual_tensor_list = []
    odour_tensor_list = []
    tensor_sizes = []


    for base_directory in session_list:

        # Create Save Directory
        output_directory = base_directory + "/Correlational_Change_Clusters/"
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        # Load Tensors
        save_directory = base_directory + "/Pre_Stimulus/"
        visual_tensor = np.load(save_directory + "Visual_Context_Correlation_Tensor.npy")
        odour_tensor = np.load(save_directory + "Odour_Context_Correlation_Tensor.npy")

        # Perform T Tests
        t_stats , p_values = stats.ttest_ind(visual_tensor, odour_tensor, axis=0)

        # Threshold t_stats
        thresholded_t_stats = np.where(p_values < 0.01, t_stats, 0)

        # View Matrix
        sorted_t_stats = sort_matrix(thresholded_t_stats)
        magnitue = np.max(np.abs(sorted_t_stats))
        plt.imshow(sorted_t_stats, vmin=-1*magnitue, vmax=magnitue, cmap='bwr')
        plt.show()

        # Cluster Matrix
        number_of_regions = np.shape(t_stats)[0]
        #number_of_clusters = 7
        model = AffinityPropagation()
        model.fit(thresholded_t_stats)

        # Get Labels
        labels = model.labels_
        number_of_clusters = np.max(labels)
        print("labels", labels)

        for cluster in range(number_of_clusters):
            cluster_indicies = []

            figure_1 = plt.figure()
            matrix_axis = figure_1.add_subplot(1,2,1)
            brain_axis = figure_1.add_subplot(1,2,2)

            for region in range(number_of_regions):
                region_label = labels[region]

                if region_label == cluster:
                    cluster_indicies.append(region)

            cluster_correlation_matrix = np.zeros(np.shape(t_stats))
            for region in cluster_indicies:
                cluster_correlation_matrix[region] = thresholded_t_stats[region]
                cluster_correlation_matrix[:, region] = thresholded_t_stats[:, region]

            sorted_cluster_correlation_matrix = sort_matrix(cluster_correlation_matrix)
            matrix_axis.imshow(sorted_cluster_correlation_matrix)
            draw_brain_network(base_directory, cluster_correlation_matrix, brain_axis)

            plt.savefig(output_directory + str(cluster).zfill(3) + ".png")
            plt.close()



session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
                "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/"]

session_list = ["/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging"]

get_correlational_changes(session_list)

