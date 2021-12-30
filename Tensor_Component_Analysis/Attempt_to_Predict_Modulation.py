import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import h5py
import tables
from scipy import signal, ndimage, stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import white_tophat
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale
from PIL import Image
import os
import cv2
import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
import sys

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


def downsample_mask(base_directory):

    # Load Mask
    mask = np.load(base_directory + "/mask.npy")

    # Downsample Mask
    original_height = np.shape(mask)[0]
    original_width = np.shape(mask)[1]
    downsampled_height = int(original_height/2)
    downsampled_width = int(original_width/2)
    downsampled_mask = cv2.resize(mask, dsize=(downsampled_width, downsampled_height))

    # Binairse Mask
    downsampled_mask = np.where(downsampled_mask > 0.1, 1, 0)
    downsampled_mask = downsampled_mask.astype(int)

    flat_mask = np.ndarray.flatten(downsampled_mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, downsampled_height, downsampled_width


def draw_brain_map(base_directory, vector, clusters_file):

    # Downsample Mask
    downsampled_indicies, downsampled_height, downsampled_width = downsample_mask(base_directory)

     # Load Clusters
    clusters = np.load(clusters_file, allow_pickle=True)
    number_of_clusters = len(clusters)

    # Draw Brain Mask
    image = np.zeros((downsampled_height * downsampled_width))
    for cluster_index in range(number_of_clusters):
        cluster = clusters[cluster_index]
        for pixel in cluster:
            pixel_index = downsampled_indicies[pixel]
            image[pixel_index] = vector[cluster_index]

    image = np.ndarray.reshape(image, (downsampled_height, downsampled_width))
    plt.imshow(image, cmap='bwr', vmin=-5, vmax=5)
    plt.show()
    #plt.savefig(base_directory + "/" + save_name + ".png")
    #plt.close()


def get_activity_tensor(activity_matrix, onsets, start_window, stop_window):

     # Create Empty Tensor To Hold Data
    activity_tensor = []

     # Get Activity Matrix For Each Trial
    for trial_onset in onsets:
        trial_start = trial_onset + start_window
        trial_stop  = trial_onset + stop_window
        trial_activity = activity_matrix[:, trial_start:trial_stop]
        trial_activity = np.transpose(trial_activity)
        activity_tensor.append(trial_activity)

    # Turn Tensor Into Array
    activity_tensor = np.array(activity_tensor)

    return activity_tensor


def get_cluster_significant_differences(base_directory, visual_responses, odour_responses, clusters_file):

    # Get Mean Activity Across Time Window
    mean_visual_response = np.mean(visual_responses, axis=1)
    mean_odour_response = np.mean(odour_responses, axis=1)

    # Perform T Test
    t_stats, p_values = stats.ttest_ind(mean_visual_response, mean_odour_response, axis=0)

    # Threshold T Stats
    threshold = 0.05
    thresholded_t_stats = np.where(p_values < threshold, t_stats, 0)

    # Draw This As A Brain Image
    draw_brain_map(base_directory, thresholded_t_stats, clusters_file)

    return t_stats, p_values


def get_mean_visual_responses(activity_matrix, visual_onsets, olfactory_onsets, start_window, stop_window):

    # Get Combined Onsets
    combined_onsets = np.concatenate([visual_onsets, olfactory_onsets])
    combined_onsets.sort()

    # Get Combined_Tensor
    combined_tensor = get_activity_tensor(activity_matrix, combined_onsets, start_window, stop_window)
    print("Combined Tensor", np.shape(combined_tensor))

    # Get Baseline and Response
    baseline = combined_tensor[:, 0:np.abs(start_window)]
    response = combined_tensor[:, np.abs(start_window):]
    print("Baseline Shape", np.shape(baseline))
    print("Response Shape", np.shape(response))


    # Get Mean Baseline And Mean Response
    mean_baseline = np.mean(baseline, axis=1)
    mean_response = np.mean(response, axis=1)
    print("Mean Baseline Shape", np.shape(mean_baseline))
    print("Mean Response Shape", np.shape(mean_response))

    normalised_response = np.divide(mean_response, mean_baseline)
    print("Response Shape", np.shape(normalised_response))

    return normalised_response


def create_correlation_tensor(activity_matrix, onsets, start_window, stop_window):

    # Get Tensor Details
    number_of_clusters = np.shape(activity_matrix)[0]
    number_of_trials = np.shape(onsets)[0]

    # Create Empty Tensor To Hold Data
    correlation_tensor = np.zeros((number_of_trials, number_of_clusters, number_of_clusters))

    # Get Correlation Matrix For Each Trial
    for trial_index in range(number_of_trials):
        print("Trial: ", trial_index, " of ", number_of_trials)

        # Get Trial Activity
        trial_start = onsets[trial_index] + start_window
        trial_stop = onsets[trial_index] + stop_window
        trial_activity = activity_matrix[:, trial_start:trial_stop]

        # Get Trial Correlation Matrix
        trial_correlation_matrix = np.corrcoef(trial_activity)

        # Add To Tensor
        correlation_tensor[trial_index] = trial_correlation_matrix

    return correlation_tensor



def get_correlations(activity_matrix, visual_onsets, olfactory_onsets):

    start_window = -75
    stop_window = -10

    # Get Combined Onsets
    combined_onsets = np.concatenate([visual_onsets, olfactory_onsets])
    combined_onsets.sort()

    # Get Correlation Tensor
    correlation_tensor = create_correlation_tensor(activity_matrix, combined_onsets, start_window, stop_window)

    return correlation_tensor


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


def perform_regression(responses, correlations):

    print(responses)
    number_of_clusters = np.shape(responses)[0]
    number_of_correlations = np.shape(correlations)[0]

    print("Number of clusters", number_of_clusters)
    print("Number of correlations", number_of_correlations)

    print("Responses shape", np.shape(responses))
    print("Correlations shape", np.shape(correlations))

    response_modulation_matrix = np.zeros((number_of_clusters, number_of_clusters))

    for response_cluster in range(number_of_clusters):
        cluster_responses = responses[response_cluster]
        for other_cluster in range(number_of_clusters):
            connectivity_values = correlations[:, response_cluster, other_cluster]
            #print(connectivity_values)
            #plt.plot(cluster_responses)
            #plt.plot(connectivity_values)
            #plt.show()
            correlation = np.corrcoef(cluster_responses, connectivity_values)[0][1]
            print(correlation)
            response_modulation_matrix[response_cluster, other_cluster] = correlation

    response_modulation_matrix = sort_matrix(response_modulation_matrix)
    plt.title("Cluster: " + str(response_cluster) + " max modulation" + str(np.max(np.abs(response_modulation_matrix))))
    plt.imshow(response_modulation_matrix, cmap='bwr', vmax=1, vmin=-1)
    plt.show()



def get_modulated_clusters(session_list):

    start_window = -10
    stop_window = 40
    clusters_file = r"/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters.npy"

    number_of_sessions = len(session_list)
    for session_index in range(number_of_sessions):

        # Load Base Directory
        base_directory = session_list[session_index]

        # Get Session Name
        session_name = base_directory.split("/")[-3]

        # Load Activity Matrix
        cluster_activity_matrix_file = base_directory + "/Cluster_Activity_Matrix.npy"
        activity_matrix = np.load(cluster_activity_matrix_file)

        # Load Stimuli Onsets
        visual_context_onsets = np.load(base_directory + r"/Stimuli_Onsets/odour_context_stable_vis_2_frame_onsets.npy")
        odour_context_onsets = np.load(base_directory + r"/Stimuli_Onsets/visual_context_stable_vis_2_frame_onsets.npy")

        print("Visual Onsets", len(visual_context_onsets))
        print("Odour onsets", len(odour_context_onsets))

        # Get Activity Tensors
        visual_activity_tensor = get_activity_tensor(activity_matrix, visual_context_onsets, start_window, stop_window)
        odour_activity_tensor = get_activity_tensor(activity_matrix, odour_context_onsets, start_window, stop_window)

        # Get Modulated Clusters
        #t_stats, p_values = get_cluster_significant_differences(base_directory, visual_activity_tensor, odour_activity_tensor, clusters_file)

        # Get Response For Each Trial
        mean_responses = get_mean_visual_responses(activity_matrix, visual_context_onsets, odour_context_onsets, start_window, stop_window)

        # Get Connections For Each Trial
        correlation_tensor = get_correlations(activity_matrix, visual_context_onsets, odour_context_onsets)

        # Reshape Tensors
        number_of_trials = np.shape(correlation_tensor)[0]
        number_of_clusters = np.shape(correlation_tensor)[1]
        #correlation_tensor = np.reshape(correlation_tensor, (number_of_trials, number_of_clusters*number_of_clusters))
        #correlation_tensor = np.transpose(correlation_tensor)
        mean_responses = np.transpose(mean_responses)
        print("Mean Response Shape", np.shape(mean_responses))
        print("cORRELATION TENSOPR", np.shape(correlation_tensor))

        perform_regression(mean_responses, correlation_tensor)

session_list = [
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging/",
    "/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging"]


get_modulated_clusters(session_list)