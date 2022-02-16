import math
import os.path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l1, l2, L1L2
from keras import layers, Sequential, Input
from keras.layers import LSTM, Dense, GRU, ZeroPadding2D
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
import time



from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LinearRegression, Ridge

import LFADS_Model_V3
import Visualise_Model
import Import_Preprocessed_Data



def load_real_data(session_list):

    low_d_data_list = []
    for session in session_list:
        low_d_data = np.load(session)
        low_d_data_list.append(low_d_data)


    return low_d_data_list


def load_inferred_data(inferred_data_directory, epoch):

    data_file = os.path.join(inferred_data_directory, str(epoch).zfill(4) + ".npy")
    data = np.load(data_file, allow_pickle=True)
    return data


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])





# Load Data
session_list = ["/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_0/Low_Dimensional_Data.npy",
                "/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_1/Low_Dimensional_Data.npy",
                "/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_2/Low_Dimensional_Data.npy",
                "/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_3/Low_Dimensional_Data.npy",
                "/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data/Session_4/Low_Dimensional_Data.npy"]

inferred_data_directory = "/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Lorentz_Example_Output_Plots"

# Load Data
number_of_sessions = len(session_list)
low_d_data_list = load_real_data(session_list)

"""
inferred_low_d_data_list = load_inferred_data(inferred_data_directory, 50000)

figure_1 = plt.figure()

inferred_axis = figure_1.add_subplot(1, 2, 1, projection='3d')
real_axis = figure_1.add_subplot(1, 2, 2, projection='3d')
colour_map = cm.get_cmap('jet')

for session_index in range(number_of_sessions):
    real_session_data = low_d_data_list[session_index]
    inferred_session_data = inferred_low_d_data_list[session_index]
    colour = colour_map(float(session_index) / number_of_sessions)

    for trajectory in real_session_data:
        real_axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=colour, alpha=0.5)

    for trajectory in inferred_session_data:
        inferred_axis.plot(trajectory[:, 2], trajectory[:, 1], trajectory[:, 0], c=colour, alpha=0.5)

    inferred_axis.view_init(elev=10, azim=10)

plt.show()
"""


plot_save_directory = r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Lorentz_example_comparison_images"

plt.ion()
# Create Figure
figure_1 = plt.figure()

rotation = 0
for epoch in range(0, 25000, 100):

    inferred_low_d_data_list = load_inferred_data(inferred_data_directory, epoch)

    inferred_axis = figure_1.add_subplot(1, 2, 1, projection='3d')
    real_axis     = figure_1.add_subplot(1, 2, 2, projection='3d')

    colour_map = cm.get_cmap('jet')

    for session_index in range(number_of_sessions):
        real_session_data = low_d_data_list[session_index]
        inferred_session_data = inferred_low_d_data_list[session_index]

        colour = colour_map(float(session_index) / number_of_sessions)

        for trajectory in real_session_data:
            real_axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=colour, alpha=0.5)

        for trajectory in inferred_session_data:
            inferred_axis.plot(trajectory[:, 2], trajectory[:, 1], trajectory[:, 0], c=colour, alpha=0.5)

    figure_1.suptitle("Epoch: " + str(epoch))

    real_axis.axis('off')
    inferred_axis.axis('off')

    real_axis.set_title("Actual Trajectories")
    inferred_axis.set_title("Inferred Trajectories")

    inferred_axis.view_init(elev=-71, azim=75 + rotation)
    real_axis.view_init(elev=4, azim=-50 + rotation)


    plt.savefig(os.path.join(plot_save_directory, str(epoch).zfill(6) + ".png"))
    plt.draw()
    plt.pause(0.1)
    plt.clf()

    rotation += 0
    if rotation > 360:
        rotation = 0

# Setup Save Directories
#plot_save_directory = r"/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Lorentz_Example_Output_Plots"