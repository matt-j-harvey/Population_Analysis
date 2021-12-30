import numpy as np
import os
import mat73


class ImportMatLabData:

    """A simple class to load the data for a single session and store it under one header
    mismatch_trials structure:
    - exp_odour: frame indices for onset / offset of expected odour - after switching from odour block to visual block, V1 trials before mouse starts licking to visual gratings. Aligned to when odour would be expected to arrive.
    - no_odour: aligned as above, but to trials in which mouse is no longer expecting odour - V2 trials at the end of a visual block, before switch
    - new_odour: positive mismatch - aligned to onset of first odour in new odour block, when mouse is not expecting an odour
    - stable_odour: aligned to last odour 1 of odour block, when mouse is reliably expecting odour
    - all_TOIs: trial indices for trials which meet the above criteria
    - which_TOIs: category of trial indexed in all_TOIs - 1 = exp_odour , 2 = no_odour , 3 = new_odour , 4 = stable_odour
    - condition_list: category labels for 4 trial types
    - switch_trial - for exp_odour trials, marks as 1 if next trial was a hit (so this trial triggered a switch), 0 if not
    - perfect_switch - for exp_odour trials, marks as 1 if trial is part of a perfect switch (i.e. misses one, then hits next 3), 0 if part of imperfect switch (more than 1 miss)
    """

    def __init__(self, matlab_path):

        self.matlab_data_path = matlab_path
        if 'preprocessed_basic.mat' not in self.matlab_data_path:
            self.matlab_data_path = os.path.join(self.matlab_data_path, 'preprocessed_basic.mat')

        big_data = mat73.loadmat(self.matlab_data_path)
        saved_mat_data = big_data['matlab_save_data']

        self.dF = saved_mat_data['dF']
        self.downsampled_AI = saved_mat_data['downsampled_AI']
        self.frame_times = saved_mat_data['frame_times']
        self.session_name = saved_mat_data['session_name']

        self.vis1_frames = saved_mat_data['vis1_trial_frames']
        self.vis2_frames = saved_mat_data['vis2_trial_frames']
        self.irrel_vis1_frames = saved_mat_data['irrel_vis1_trial_frames']
        self.irrel_vis2_frames = saved_mat_data['irrel_vis2_trial_frames']
        self.odr1_frames = saved_mat_data['odr1_trial_frames']
        self.odr2_frames = saved_mat_data['odr2_trial_frames']

        self.trial_type_data = saved_mat_data['trial_type_data']
        self.mismatch_trials = saved_mat_data['mismatch_trials']

        #print('Data imported.')