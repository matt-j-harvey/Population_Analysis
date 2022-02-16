import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.gridspec as gridspec

from mvlearn.embed import GCCA

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Create_Activity_Tensor
import Widefield_General_Functions


controls = ["/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

controls =  ["/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
             "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
             "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
             ]

matrix_list = []



#Load Mean Responses
tensor_list = []
for base_directory in controls:

    # Load Activity Tensors
    activity_tensor_directory = os.path.join(base_directory, "Activity_Tensors")
    vis_1_visual_tensor = np.load(os.path.join(activity_tensor_directory, "Vis_1_Stable_Visual" + "_Activity_Tensor.npy"))
    vis_1_visual_tensor = np.nan_to_num(vis_1_visual_tensor)
    vis_1_visual_tensor = np.mean(vis_1_visual_tensor, axis=0)

    vis_2_visual_tensor = np.load(os.path.join(activity_tensor_directory, "Vis_2_Stable_Visual" + "_Activity_Tensor.npy"))
    vis_2_visual_tensor = np.nan_to_num(vis_2_visual_tensor)
    vis_2_visual_tensor = np.mean(vis_2_visual_tensor, axis=0)

    """
    vis_1_odour_tensor = np.load(os.path.join(activity_tensor_directory, "Vis_1_Stable_Odour" + "_Activity_Tensor.npy"))
    vis_1_odour_tensor = np.nan_to_num(vis_1_odour_tensor)
    vis_1_odour_tensor = np.mean(vis_1_odour_tensor, axis=0)

    vis_2_odour_tensor = np.load(os.path.join(activity_tensor_directory, "Vis_2_Stable_Odour" + "_Activity_Tensor.npy"))
    vis_2_odour_tensor = np.nan_to_num(vis_2_odour_tensor)
    vis_2_odour_tensor = np.mean(vis_2_odour_tensor, axis=0)
    """

    #concatenate
    combined_tensor = np.concatenate([vis_1_visual_tensor, vis_2_visual_tensor])#, vis_1_odour_tensor, vis_2_odour_tensor])
    print("Combined Tensor Shape", np.shape(combined_tensor))
    tensor_list.append(combined_tensor)


number_of_components = 5
number_of_sessions = len(controls)
gcca = GCCA(n_components=number_of_components)

# Fit Model and Transform Data
transformed_data = gcca.fit_transform(tensor_list)

"""
# Plot Traces
print("Transformed Data Shape", transformed_data.shape)
for component_index in range(number_of_components):
    for session_index in range(number_of_sessions):
        plt.plot(transformed_data[session_index][:, component_index])
    plt.show()
"""

# Get Projection Matricies
projection_matrixies = gcca.projection_mats_
print("Projeciton Matrix Shape", np.shape(projection_matrixies))


indicies, image_height, image_width = Widefield_General_Functions.load_mask(controls[0])


figure_1 = plt.figure()
figure_1_gridspec = gridspec.GridSpec(nrows=number_of_sessions, ncols=number_of_components, figure=figure_1)

n_rows = number_of_components
n_columns = number_of_sessions

for component_index in range(number_of_components):
    for session_index in range(number_of_sessions):
        axis = figure_1.add_subplot(figure_1_gridspec[session_index, component_index])
        axis.axis('off')
        projection_vector = projection_matrixies[session_index][:, component_index]
        projection_image = Widefield_General_Functions.create_image_from_data(projection_vector, indicies, image_height, image_width)
        axis.imshow(projection_image, vmin=0, cmap='inferno')


plt.show()