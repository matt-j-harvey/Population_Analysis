import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
import os
from scipy.io import loadmat, savemat
import mat73
import pandas as pd
from mvlearn.embed import GCCA
from sklearn.cluster import OPTICS, AffinityPropagation, DBSCAN
import jPCA
from jPCA.util import load_churchland_data, plot_projections
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def plot_gcca_mean_traces(transformed_data, number_of_sessions, number_of_components, unique_condition_labels, condition_length, number_of_conditions, number_of_timepoints, opsin_list):

    cmap = cm.get_cmap('tab10')
    cmap_opsin = cm.get_cmap('Pastel1')
    unique_opsins = np.unique(opsin_list)

    for component_index in range(number_of_components):

        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1,1,1)
        colour_list = []

        for session_index in range(number_of_sessions):

            component_trace = transformed_data[session_index][:, component_index]
            found_opsin = np.argwhere(unique_opsins == opsin_list[session_index])
            opsin_colour = cmap_opsin(found_opsin[0])[0]
            axis_1.plot(component_trace, color=opsin_colour)

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


def plot_clustering_labels(proj_matrices, session_data, n_components=4, method='OPTICS'):

    for session in range(len(proj_matrices)):
        cluster_data = proj_matrices[session][:, 0:n_components]
        if method == 'OPTICS':
            cluster_labels = OPTICS(min_samples=5).fit_predict(X=cluster_data)
        elif method == 'DBSCAN':
            cluster_labels = DBSCAN(min_samples=5).fit_predict(cluster_data)
        elif method == 'Affinity':
            cluster_labels = AffinityPropagation(random_state=5).fit_predict(cluster_data)

        pd_labelled = pd.DataFrame(cluster_data)
        pd_labelled['label'] = cluster_labels

        #sns.jointplot(data=pd_labelled, x=0, y=1, hue='label')
        sns.pairplot(data=pd_labelled, hue='label')


def reshape_neural_data(neural_data_list):

    transformed_data = []

    for raster in neural_data_list:
        #dropped_raster = raster[0]
        transposed_raster = np.transpose(raster)
        transposed_raster = np.nan_to_num(transposed_raster)
        transformed_data.append(transposed_raster)

    return transformed_data


def filter_matlab_data(matlab_data, selected_opsin, selected_laser, filter_vip=False):

    number_of_sessions = len(matlab_data['neural_data'])
    opsin_list = matlab_data['opsin']

    # Solve the problem of some variables being stuck in a sublist
    if isinstance(matlab_data['neural_data'][0], list):
        matlab_data['VIP_index'] = [x[0] for x in matlab_data['VIP_index']]
        #matlab_data['opsin'] = [x[0] for x in matlab_data['opsin']]
        matlab_data['neural_data'] = [x[0] for x in matlab_data['neural_data']]

    laser_index_check = matlab_data['laser_data'][0, :] == selected_laser
    vis_index_check = ['odr' not in x for x in matlab_data['stim_data'][0]]
    timepoint_length = np.sum(np.logical_and(laser_index_check, vis_index_check))

    if selected_opsin == 'all':
        opsin_index = range(number_of_sessions)
    else:
        opsin_list = [x[0] == selected_opsin for x in opsin_list]
        opsin_index = np.nonzero(opsin_list)
        opsin_index = opsin_index[0]

    filtered_data = {'VIP_index': [],
                     'cond_labels': [],
                     'laser_data': np.empty((len(opsin_index), timepoint_length)),
                     'mouse_name': [],
                     'neural_data': [],
                     'opsin': [],
                     'recording_site': [],
                     'session_name': [],
                     'stim_data': [],
                     'times': np.empty((len(opsin_index), timepoint_length))
                     }

    cc = 0
    for session_index in opsin_index:
        laser_index = matlab_data['laser_data'][session_index, :] == selected_laser
        vis_index = ['odr' not in x for x in matlab_data['stim_data'][session_index]]
        timepoint_filter = np.logical_and(laser_index, vis_index)

        for key in matlab_data.keys():
            if isinstance(matlab_data[key], list):

                data_shape = np.shape(matlab_data[key][session_index])
                if len(data_shape) == 1 and data_shape[0] == len(timepoint_filter):
                    temp_data = np.array(matlab_data[key][session_index])
                    filtered_data[key].append(temp_data[timepoint_filter])

                elif len(data_shape) == 1:
                    filtered_data[key].append(matlab_data[key][session_index])

                elif len(data_shape) == 2:
                    if data_shape[0] == len(timepoint_filter):
                        filtered_data[key].append(matlab_data[key][session_index][timepoint_filter, :])
                    elif data_shape[1] == len(timepoint_filter):
                        if filter_vip:
                            data_to_filter = matlab_data[key][session_index]
                            data_to_filter = data_to_filter[np.logical_not(matlab_data['VIP_index'][session_index]), :]
                            #data_to_filter = data_to_filter[matlab_data['VIP_index'][session_index], :]
                            data_to_filter = data_to_filter[:, timepoint_filter]
                            filtered_data[key].append(data_to_filter)
                        else:
                            filtered_data[key].append(matlab_data[key][session_index][:, timepoint_filter])

            elif isinstance(matlab_data[key], np.ndarray):
                filtered_data[key][cc, :] = matlab_data[key][session_index, timepoint_filter]

        cc += 1

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
file_location = r"E:\Data\GCCA_export\Dylan_all_VIP_mice_GCCA_data.mat"

# Load Matlab Data
matlab_data = mat73.loadmat(file_location)
matlab_data = matlab_data['output_GCCA']

# Filter by opsin, laser and cell type
filtered_data = filter_matlab_data(matlab_data, selected_opsin='ArchT', selected_laser=0, filter_vip=True)

filtered_opsins = filtered_data['opsin']

# Get condition labels
unique_condition_labels, condition_length, number_of_conditions, number_of_timepoints = load_condition_labels(filtered_data)

# Load and Transform Neural Data
neural_data = filtered_data['neural_data']
neural_data = reshape_neural_data(neural_data)

# Perform GCCA
number_of_sessions = len(neural_data)
number_of_components = 20
gcca_model =GCCA(n_components=number_of_components)
transformed_data = gcca_model.fit_transform(neural_data)

laser_data = filter_matlab_data(matlab_data, selected_opsin='ArchT', selected_laser=100, filter_vip=True)
laser_condition_labels, laser_condition_length, number_of_laser_conditions, number_of_laser_timepoints = load_condition_labels(laser_data)
laser_neural_data = laser_data['neural_data']
laser_neural_data = reshape_neural_data(laser_neural_data)
projected_laser_data = gcca_model.transform(laser_neural_data)

'''
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Neural Net",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    MLPClassifier(alpha=1, max_iter=1000),
]


laser_data['opsin'] = [x[0] for x in laser_data['opsin']]
uniqueOpsins = list(set(laser_data['opsin']))
opsin_Id = [uniqueOpsins.index(x) for x in laser_data['opsin']]
opsin_Id = np.array(opsin_Id)

performance = np.empty([len(classifiers), np.shape(projected_laser_data)[1]])
# loop through classifier
counter = 0
for name, clf in zip(names, classifiers):
    print('Fitting classifier:' + name)
    # Loop through temporal components
# for cc in range(5):

    # Loop through timepoint
    for tt in range(np.shape(projected_laser_data)[1]):
        selectData = projected_laser_data[:, tt, [0, 1, 2]]

        # leave one out cross validation
        loo = LeaveOneOut()
        cv_scores = []
        for train_index, test_index in loo.split(selectData):
            clf.fit(selectData[train_index, :], opsin_Id[train_index])
            cv_scores.append(clf.score(selectData[test_index, :], opsin_Id[test_index]))

        performance[counter, tt] = np.mean(cv_scores)

    counter += 1

# Plot classifier performance
cmap = cm.get_cmap('tab10')
clf_map = cm.get_cmap('Set2')
figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1, 1, 1)
colour_list = []

line_list = []
for cc in range(len(classifiers)):
    clf_line = axis_1.plot(performance[cc, :], color=clf_map(cc/len(classifiers)), label=names[cc])
    line_list.append(clf_line)

plt.axhline(y=0.33, color='k', linestyle=':')

# Draw Black Lines To Demarcate Conditions
for x in range(0, number_of_timepoints, condition_length):
    axis_1.axvline(x, ymin=-1, ymax=1, color='k')

# Shade Conditions
for condition_index in range(0, number_of_conditions):
    condition_colour = cmap(float(condition_index) / number_of_conditions)
    colour_list.append(condition_colour)

    start = condition_index * condition_length
    stop = start + condition_length

    axis_1.axvspan(xmin=start, xmax=stop - 1, facecolor=condition_colour, alpha=0.3)

# Add Legengs
patch_list = []
for condition_index in range(number_of_conditions):
    patch = mpatches.Patch(color=colour_list[condition_index], label=unique_condition_labels[condition_index],
                                   alpha=0.3)
    patch_list.append(patch)

line_list = [x[0] for x in line_list]
axis_1.legend(handles=line_list)
plt.ylabel('Decoding performance')

plt.show()
'''
#output_dict = {'GCCA_matrix': gcca_model.projection_mats_}
#savemat(r"E:\Data\GCCA_export\GCCA_projection_matrices.mat", output_dict)


#plot_clustering_labels(gcca_model.projection_mats_, filtered_data, n_components=2, method='OPTICS')

# Plot GCCA Data
plot_gcca_mean_traces(transformed_data, number_of_sessions, number_of_components, unique_condition_labels, condition_length, number_of_conditions, number_of_timepoints, filtered_opsins)

# Plot projected laser data
plot_gcca_mean_traces(projected_laser_data, number_of_sessions, number_of_components, laser_condition_labels, laser_condition_length, number_of_laser_conditions, number_of_laser_timepoints, filtered_opsins)


'''
# Load publicly available data from Mark Churchland's group
path = "/Users/Bantin/Documents/Stanford/Linderman-Shenoy/jPCA_ForDistribution/exampleData.mat"
datas, times = load_churchland_data(path)

# Create a jPCA object
jpca = jPCA.JPCA(num_jpcs=2)

# Fit the jPCA object to data
(projected,
 full_data_var,
 pca_var_capt,
 jpca_var_capt) = jpca.fit(datas, times=times, tstart=-1, tend=1)

# Plot the projected data
plot_projections(projected)

'''