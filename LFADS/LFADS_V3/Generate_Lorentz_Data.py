import os.path
import numpy as np
import matplotlib.pyplot as plt



def get_lorentz_trajectory(initial_position, number_of_timepoints=100, timestep=0.01):

    # Lorentz Attractor Parameters
    alpha = 10
    beta = float(8) / 3
    rho = 28

    # Create Empty Array To Hold The Trajectory
    trajectory = np.zeros((number_of_timepoints, 3))
    trajectory[0] = initial_position

    # Setup Initial Positions
    x_pos = initial_position[0]
    y_pos = initial_position[1]
    z_pos = initial_position[2]

    for timepoint in range(1, number_of_timepoints):

        # Calculate Derivatives
        dx = alpha * (y_pos - x_pos)
        dy = x_pos * (rho - z_pos) - y_pos
        dz = (x_pos * y_pos) - (beta * z_pos)

        # Scale Derivatives By Timestep
        dx = dx * timestep
        dy = dy * timestep
        dz = dz * timestep

        # Add Deritvatives To Current Positions
        x_pos += dx
        y_pos += dy
        z_pos += dz

        # Add this New Point To The Trajectory
        trajectory[timepoint] = [x_pos, y_pos, z_pos]

    return trajectory


def normalise_data(data):

    # Get Data Shape
    number_of_trials     = np.shape(data)[0]
    number_of_timepoints = np.shape(data)[1]
    number_of_dimensions = np.shape(data)[2]

    # Reshape Data
    data = np.ndarray.reshape(data, (number_of_trials * number_of_timepoints, number_of_dimensions))

    # Divide By Max Values
    abs_data = np.abs(data)
    max_vector = np.max(abs_data, axis=0)
    data = np.divide(data, max_vector)

    # Put Back Into Original Shape
    data = np.ndarray.reshape(data, (number_of_trials, number_of_timepoints, number_of_dimensions))

    return data




def generate_pseuodata(save_directory, number_of_trials=5, number_of_neurons=25, number_of_timepoints=100, initial_position_range=25, plot=False):

    # Create Neuron Matrix
    neuron_matrix = np.random.normal(0, 1, (number_of_neurons, 3))

    # Get Low Dimensional Trajectories
    low_diemensional_data = np.zeros((number_of_trials, number_of_timepoints, 3))
    for trial_index in range(number_of_trials):
        initial_position = np.random.uniform(low=-1 * initial_position_range, high=initial_position_range, size=3)
        trajectory = get_lorentz_trajectory(initial_position, number_of_timepoints=number_of_timepoints)
        low_diemensional_data[trial_index] = trajectory

    # Normalise Low D Trajectories
    low_diemensional_data = normalise_data(low_diemensional_data)

    # Create Neural Activity By Multiplying Neuron Matricies and Low Dimensional Trajectories
    high_dimensional_data = np.zeros((number_of_trials, number_of_timepoints, number_of_neurons))
    for trial_index in range(number_of_trials):
        for timepoint in range(number_of_timepoints):
            neural_activity = np.dot(neuron_matrix, low_diemensional_data[trial_index][timepoint])
            high_dimensional_data[trial_index][timepoint] = neural_activity

    # Plot Trajectories
    plot_low_dimensional_trajectories(low_diemensional_data)

    # Save This Data
    print("Data Shape", np.shape(high_dimensional_data))
    np.save(os.path.join(save_directory,  "Low_Dimensional_Data.npy"),   low_diemensional_data)
    np.save(os.path.join(save_directory,  "High_Dimensional_Data.npy"), high_dimensional_data)
    np.save(os.path.join(save_directory, "Neuron_Matrix.npy"),         neuron_matrix)



def plot_low_dimensional_trajectories(trial_data):
    ax = plt.axes(projection='3d')
    for trial in trial_data:
        ax.plot(trial[:, 0], trial[:, 1], trial[:, 2], alpha=0.4)
    plt.show()








# Generate Trial Data
number_of_sessions = 5
for session in range(number_of_sessions):

    number_of_neurons = np.random.randint(low=80, high=120)
    number_of_trials = np.random.randint(low=20, high=35)

    save_directory = os.path.join("/home/matthew/Documents/Github_Code/Population_Analysis/LFADS/LFADS_V3/Pseudo_data", "Session_" + str(session))
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    generate_pseuodata(save_directory, number_of_trials=number_of_trials, number_of_neurons=number_of_neurons, number_of_timepoints=50, plot=True)




