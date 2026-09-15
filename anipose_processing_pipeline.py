#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:11:23 2023

@author: nathanieltse
"""

"""
Input: Anipose-type folder with frame-synced 2d tracking completed. 
Performs: 
    Creates empty duplicate folder and copies over filtered .csv files and pickles 
    Creates .h5 files from filtered .csv files
    FPS matching (assuming cam1 is the Jetson cam)
    Splits 2D tracking into minute chunks for triangulation
    Detects reconstruction errors 
        Further splits erroneous minute chunk(s) into second chunks
        Finds problematic second(s)
        Ablates problematic second(s) in corresponding minute chunk(s)
        Re-runs triangulation on now fixed minute chunk(s)
    Stitches triangulated minute chunks into full-length 3D tracking files
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import matplotlib
import matplotlib.animation
import argparse

def create_project_clone(anipose_folder, filtered_data = True):
    """Duplicate the whole project structure, transfering each trial's filtered csv and pickles"""
    os.chdir(anipose_folder)
    trials = os.listdir()
    if not 'config.toml' in trials:
        raise Exception('config.toml not detected in this project folder, so nothing was created.')
    trials = [i for i in trials if (os.path.isdir(i) and not i.startswith('temp'))]
    if not trials:
        raise Exception('No trials detected in this project folder so nothing was created. Check your project path')
    for trial in trials:
        real_2d_pose = os.path.join(trial, 'pose-2d')
        temp_2d_pose = os.path.join('temp_project', f'temp_{trial}', 'pose-2d')
        os.makedirs(temp_2d_pose, exist_ok = True)
        transfer_files = os.listdir(real_2d_pose)
        def check_file(file, filtered_data):
            """If file is a csv (and either contains filtered or non-filtered data), return full path"""
            if filtered_data:
                if file.endswith(('filtered.csv', 'pickle')):
                    return os.path.join(real_2d_pose, file)
            else:
                if 'filtered' not in file and file.endswith(('csv', 'pickle')):
                    return os.path.join(real_2d_pose, file)
        transfer_files = [check_file(i, filtered_data) for i in transfer_files if check_file(i, filtered_data)]
        for file in transfer_files:
            shutil.copy(file, temp_2d_pose)
        try:
            shutil.copytree(os.path.join(trial, 'calibration'), os.path.join('temp_project', f'temp_{trial}', 'calibration'))
        except FileExistsError:
            print(f'Did you know you already had a calibration folder in the temp_project folder for {trial}?\nWe left it alone')
    shutil.copy('config.toml', 'temp_project')
    
def correct_framerate(file, fps_adjustment = 0.994459834):
    """Chop down cam1 2D tracking to match FPS of other three videos
    
    Input: cam1 csv file
    
    A 10 minute video recorded on our laptop contains 17950 ± 1 frames (29.917 fps)
    A 10 minute video recorded on our Jetson Nano contains 18050 ± 1 frames (30.083 fps)
    So to correct for this difference, we need to reduce frames in the cam1 tracking 
        by a factor of 0.994459834
    If you want to use FFMPEG to change the video fps (and suffer the myriad issues with re-encoding)
        multiply the alleged fps (30) by the conversion factor and enter that as your desired new fps:
        30 * 0.994459834 = 29.83379501
    """
    cc = pd.read_csv(file, header = None, dtype = object)
    if not type(cc.iloc[0,0]) == str:
        assert np.isnan(cc.iloc[0,0])
        cc = pd.read_csv(file, header = 0, index_col = 0)       
    cc_ = pd.DataFrame(columns = cc.iloc[:3, 0])
    cc_['scorer'] = cc.iloc[0, 1:]
    cc_['bodyparts'] = cc.iloc[1, 1:]
    cc_['coords'] = cc.iloc[2, 1:]
    cc_ind = pd.MultiIndex.from_frame(cc_)
    dd = pd.DataFrame(cc.iloc[3:, 1:].values, columns = cc_ind)
    to_omit = len(dd) * (1 - fps_adjustment)
    to_omit = len(dd) / to_omit
    range_to_omit = range(0, len(dd), int(to_omit))
    hdf_dropped = dd.drop(range_to_omit).reset_index(drop=True)
    csv_dropped = pd.concat([cc.iloc[:3], 
                            cc.iloc[3:].reset_index(drop=True).drop(range_to_omit)]).reset_index(drop=True)
    return csv_dropped, hdf_dropped

def convert_filtered_csv(file):
    """Create HDF files from csv files
    
    DeepLabCut may output filtered .h5 files that are not actually filtered;
    as such, we always take csv files and directly make them into .h5 files
    for use later. Even when using unfiltered data, we create h5 files from csv to keep
    things consistent.
    """
    cc = pd.read_csv(file, header = None, dtype = object)
    cc_ = pd.DataFrame(columns = cc.iloc[:3, 0])
    cc_['scorer'] = cc.iloc[0, 1:]
    cc_['bodyparts'] = cc.iloc[1, 1:]
    cc_['coords'] = cc.iloc[2, 1:]
    cc_ind = pd.MultiIndex.from_frame(cc_)
    dd = pd.DataFrame(cc.iloc[3:, 1:].values, columns = cc_ind).astype(np.float64)
    return dd  

def get_temp_trials(anipose_folder, overwrite):
    """Return absolute paths to all temp_project directories"""
    temp_path = os.path.join(anipose_folder, 'temp_project')
    try:
        trials = os.listdir(temp_path)
    except FileNotFoundError:
        raise Exception('Failed to find a temp_project folder - check project path or run create_project_clone first')
    trials = [os.path.join(temp_path, i) for i in trials]
    trials = [i for i in trials if os.path.isdir(i)]
    if not overwrite:
        trials_ = []
        for trial in trials:
            if os.path.isdir(os.path.join(trial, 'pose-3d')):
                if any([i.endswith('copyfile.csv') for i in os.listdir(os.path.join(trial, 'pose-3d'))]):
                    print(f'{trial} seems complete already, skipping for now...')
                    continue
            trials_.append(trial)
    if not trials:
        raise Exception('No trials detected in this temp_project folder so nothing was changed')
    return trials

def get_prefix(file_name):
    """Attempt to recover a path minus the file extension"""
    prefix = file_name.split('.')[0]
    if len(file_name.split('.')) > 2:
        raise Exception(f'Ran into an issue extracting a file name; is there more than one period in the path {file_name}?')
    else:
        return prefix

def trim_to_shortest(cam_csvs):
    """Just by nature of how long it takes the cameras to start recording and how the sync signal is timed, some videos
    can be considerably different in length than others. As a result, when splitting into minutes, we can accidentally
    produce an unequal number of minute-files per cam if two or more camera recordings' durations are different by more
    than a minute

    Therefore, we need to find the shortest video, and cut all other videos to that length - with the added bonus that
    we can guarantee that every included frame has views from all cameras, rather than the number of views dropping off
    as we approach the end of the video

    All takes place in the temp_project folder
    """

    cam_lengths = []
    for cam in cam_csvs:
        df = pd.read_csv(cam, header = None)
        cam_lengths.append(len(df))
    shortest = np.nanmin(cam_lengths)
    for cam in cam_csvs:
        df = pd.read_csv(cam, header = None, index_col = 0, dtype = object)
        new_df = df.iloc[:shortest, :]
        os.remove(cam)
        new_df.to_csv(cam, header = None, index_label = '0')

def project_preprocess(trial, overwrite = True, truncate = None):
    """Correct cam1 frame rate, create .h5 files from filtered csvs
    
    When testing different parameters we may want to use only a subsection of the 
    2D tracking data (especially if creating/storing large video files). To achieve this,
    set the truncate variable to a tuple of (start,stop) where start and stop are frame numbers.
    """
    local_2d = os.path.join(trial, 'pose-2d')
    files = os.listdir(local_2d)
    if 'correct_fps.txt' not in files:
        cam1_file = [os.path.join(local_2d, i) for i in files if re.search('cam1', i) and not i.endswith('pickle')][0]
        #print(cam1_file)
        csv, hdf = correct_framerate(cam1_file)
        csv.to_csv(cam1_file, index = False, header = False)
        Path(os.path.join(local_2d, 'correct_fps.txt')).touch()
    else:
        print(f'already fixed cam1 fps for this trial: {trial}, skipping and moving on')
    cam_csvs = [os.path.join(local_2d, i) for i in files if i.endswith('csv') and not i.startswith('._')]
    #df = pd.read_csv(cam_csvs[2], header = None)
    #print(df)
    trim_to_shortest(cam_csvs)
    #df = pd.read_csv(cam_csvs[0], header = None)
    #print(df)
    for cam in cam_csvs:
        converted = convert_filtered_csv(cam)
        prefix = get_prefix(cam)
        if os.path.isfile(prefix + '_copyfile.h5') and overwrite == False:
            print('An hdf file is already present; skipping overwriting it')
            os.remove(cam)
            continue
        if truncate:
            start,stop = truncate
            converted = converted.iloc[start:stop].reset_index(drop=True)
        converted.to_hdf(prefix + '_copyfile.h5', 'f')
        os.remove(cam)

def split_to_minutes(trial):
    """Split csv/h5 files to minute-length chunks assuming 30fps
    
    Input must be a temp trial directory
    """
    local_2d = os.path.join(trial, 'pose-2d')
    files = [os.path.join(local_2d, i) for i in os.listdir(local_2d) if i.endswith('_copyfile.h5')]
    for file in files:
        data = pd.read_hdf(file)
        prefix = get_prefix(file)
        interval = 0
        while True:
            start = 0 + (1800 * interval)
            end = 1800 + (1800 * interval)
            if end > len(data):
                interval_data = data.iloc[start:]
                if interval_data.empty:
                    break
                interval_data.to_hdf(prefix + '_' + str(interval) + '.h5', 'f')
                break
            interval_data = data.iloc[start:end]
            interval_data.to_hdf(prefix + '_' + str(interval) + '.h5', 'f')
            interval += 1
        os.remove(file)

def find_lims(data):
    """Useful for finding best 3d limits to use for plotting animations but
    also a quick way to find messed up triangulations, as these limits will generally
    be much larger. x and y limits tend to be between -10 and 10, and z is generally 4 to 25.
    Any limit with absolute value greater than 50 is probably a result of bad triangulation
    """
    columns = [i for i in data.columns if '_' in i]
    columns = [i for i in columns if i.split('_')[-1] in ['x','y','z']]
    xs = [i for i in columns if 'x' in i]
    ys = [i for i in columns if 'y' in i]
    zs = [i for i in columns if 'z' in i]
    xlims = (data.loc[:,xs].min().min(), data.loc[:,xs].max().max())
    ylims = (data.loc[:,ys].min().min(), data.loc[:,ys].max().max())
    zlims = (data.loc[:,zs].min().min(), data.loc[:,zs].max().max())
    lim_list = []
    for lims in [xlims, ylims, zlims]:
        #print(lims)
        lim_list.append(max([abs(i) for i in lims]))
    return lim_list

def find_outliers(df):
    """Find the IQR for a dataset and use to find outliers"""
    q3 = df.quantile(.75)
    q1 = df.quantile(.25)
    iqr = q3 - q1
    upper_thresh = q3 + (1.5 * iqr)
    return df > upper_thresh

def check_errors(trial, time_frame = 'minutes', method = 'absolute', feedback = False, strictness = 'low'):
    """Identify outlier limits indicating failed reconstruction
    
    Input must be an absolute path to a trial. If time_frame is minutes, we just report
    which minutes are bad. But if time_frame is seconds, we return a string corresponding
    to a minute and its second that is bad
    """
    local_3d = os.path.join(trial, 'pose-3d')
    files = os.listdir(local_3d)
    files = [i for i in files if i.endswith('csv')]
    files = [i for i in files if re.search('copyfile_.+csv', i)]
    if time_frame == 'minutes':
        files = [i for i in files if 'second' not in i]
        files = sorted(files, key = lambda x: int(re.search('copyfile_(.+).csv', x)[1]))
    elif time_frame == 'seconds':
        files = [i for i in files if 'second' in i]
        files = sorted(files, key = lambda x: int(re.search('second_(.+).csv', x)[1]))
    else:
        raise Exception('Invalid time_frame')
    all_intervals = []
    interval_lims = []
    for file in files:
        if time_frame == 'minutes':
            interval_number = re.search('copyfile_(.+).csv', file)[1]
        elif time_frame == 'seconds':
            temp = re.search('copyfile_(.+)_second_(.+).csv', file)
            interval_number = ' '.join([temp[1], temp[2]])
        data = pd.read_csv(os.path.join(local_3d,file))
        all_intervals.append(data)
        interval_lims.append(pd.Series(find_lims(data), name = interval_number))
    if time_frame == 'seconds':
        interval_lims = sorted(interval_lims, key = lambda x: int(x.name.split(' ')[1]))
        interval_lims = sorted(interval_lims, key = lambda x: int(x.name.split(' ')[0]))
    interval_lims = pd.concat(interval_lims, axis = 1)
    interval_lims.index = ['x','y','z']
    interval_lims.T.plot.bar()
    plt.suptitle('x and y are usually below 10, z is under 25')
    if method == 'absolute':
        if strictness == 'high':
            outliers = (interval_lims.T > [15, 15, 45]).any(axis=1)
            outliers_strict = (interval_lims.T > [10, 10, 25]).all(axis=1)
            outliers = outliers | outliers_strict
        else:
            outliers = (interval_lims.T > [15, 15, 45]).any(axis=1)
    elif method == 'iqr':
        outliers = find_outliers(interval_lims.T).any(axis=1)
    else:
        outliers = (interval_lims.loc['z'] == interval_lims.loc['z'].max()).T
    if any(outliers):
        minute_chunks = outliers.index[np.where(outliers)[0]]
        if feedback:
            print(f'We had {len(minute_chunks)} problematic minutes')
            choice = input(f'If you wish to skip these, input "SKIP"\n>')
            if choice == 'SKIP':
                print('what the fuck')
                return None
        return minute_chunks
    else:
        return None
    
def split_to_seconds(trial, bad_chunks):
    """Split bad minutes up into seconds and place into a new directory for another round of triangulation
    
    Input is an absolute path to a temp_trial
    """
    local_2d = os.path.join(trial, 'pose-2d')
    files = os.listdir(local_2d)
    files = [os.path.join(local_2d, i) for i in files if i.endswith('.h5')]
    files = sorted(files, key = lambda x: int(re.search('copyfile_(.+).h5', x)[1]))
    for file in files:
        interval_number = re.search('copyfile_(.+).h5', file)[1]
        if interval_number not in bad_chunks:
            continue
        data = pd.read_hdf(file)
        prefix = get_prefix(file)
        interval = 0
        while True:
            start = 0 + (30 * interval)
            end = 30 + (30 * interval)
            if end > len(data):
                interval_data = data.iloc[start:]
                if len(interval_data) < 1:
                    break                    
                interval_data.to_hdf(prefix + '_second_' + str(interval) + '.h5', 'f')
                break
            else:
                interval_data = data.iloc[start:end]
            interval_data.to_hdf(prefix + '_second_' + str(interval) + '.h5', 'f')
            interval += 1

def fix_bad_chunks(trial, bad_seconds):
    """Take a minute and second, find that minute and second and wipe clear"""
    local_2d = os.path.join(trial, 'pose-2d')
    files = os.listdir(local_2d)
    files = [os.path.join(local_2d, i) for i in files if 'second' not in i]
    for bad in bad_seconds:
        bad_minute, bad_second = bad.split(' ')
        cam_files = [i for i in files if f'copyfile_{bad_minute}.h5' in i]
        assert len(cam_files) == 4, f'unequal number of cam tracks for minute {bad}, instead got{[print(i) for i in cam_files]}'
        for cam in cam_files:
            data = pd.read_hdf(cam)
            start = int(bad_second) * 30
            stop = start + 30
            data.iloc[start:stop] = np.nan
            os.remove(cam)
            data.to_hdf(cam, 'f')
        
def clean_up(trial, bad_minutes = []):
    """Delete now-useless 2D second chunks and all 3D chunks so we can re-triangulate
    Or only delete the 3D chunks from minutes that were problematic to avoid redoing everything
    To do the latter, bad_minutes needs to be a list or index of bad minutes"""
    local_2d = os.path.join(trial, 'pose-2d')
    files = os.listdir(local_2d)
    files = [os.path.join(local_2d, i) for i in files if 'second' in i]
    for file in files:
        os.remove(file)
    local_3d = os.path.join(trial, 'pose-3d')
    files = os.listdir(local_3d)
    for file in files:
        if '_second_' in file:
            os.remove(os.path.join(local_3d, file))
            continue
        if len(bad_minutes) > 0:
            if any (f'copyfile_{bad_minute}.csv' in file for bad_minute in bad_minutes):
                os.remove(os.path.join(local_3d, file))
        else:
            os.remove(os.path.join(local_3d, file))
    
def stitch_minutes(trial, show_result = False):
    """Stick this shit in a seperate module holy shit"""
    def get_data(bp_name, frame):
        bp_cords = [bp_name + '_x', bp_name + '_y', bp_name + '_z']
        bp_cords = frame[bp_cords]
        return bp_cords
    def show_frame(frame, ax, threed = True): 
        ax.set_prop_cycle(color=['red', 'green', 'blue'])
        if threed:
            all_syss = []
            for limbsys in [backbone, front_limbs, back_limbs, ears]:
                parts = []
                for bodypart in limbsys:
                    x,y,z = get_data(bodypart, frame)
                    #ax.scatter(x,y,z, alpha = .5)
                    parts.append([x,y,z])
                ax.plot([i[0] for i in parts],
                        [i[1] for i in parts],
                        [i[2] for i in parts],
                        alpha = .5)
    def update(i):
        ax.clear()
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
        show_frame(data.loc[i], ax)
    bodyparts = ["nose", 
                "neck", 
                "half_neck", 
                "center", 
                "half_tailbase", 
                "tailbase", 
                "tailtip", 
                "l_ear_mousper", 
                "r_ear_mouseper", 
                "LBack_paw", 
                "LBack_root", 
                "RBack_paw", 
                "RBack_root", 
                "RF_paw", 
                "RF_root",
                "LF_paw", 
                "LF_root"]
    backbone = ["nose", 
                "neck", 
                "half_neck", 
                "center", 
                "half_tailbase", 
                "tailbase", 
                "tailtip"]
    front_limbs = ["RF_paw", 
                   "RF_root",
                   "half_neck",
                   "LF_root", 
                   "LF_paw"]
    back_limbs = ["LBack_paw", 
                  "LBack_root", 
                  "tailbase",
                  "RBack_root", 
                  "RBack_paw"]
    ears = ['l_ear_mousper',
            'nose',
            'r_ear_mouseper']
    local_3d = os.path.join(trial, 'pose-3d')
    files = os.listdir(local_3d)
    files = [i for i in files if i.endswith('csv')]
    files = [i for i in files if re.search('copyfile_.+csv', i)]
    files = sorted(files, key = lambda x: int(re.search('copyfile_(.+).csv', x)[1]))
    all_intervals = []
    for file in files:
        data = pd.read_csv(os.path.join(local_3d,file))
        all_intervals.append(data)
    data = pd.concat(all_intervals, axis = 0).reset_index(drop=True)
    group_name = '_'.join(file.split('_')[:-1]) + '.csv'
    data.to_csv(os.path.join(local_3d, group_name), index = False) # Results in a data file ending with 'copyfile.csv'
    if show_result:
        columns = [i for i in data.columns if '_' in i]
        columns = [i for i in columns if i.split('_')[-1] in ['x','y','z']]
        xs = [i for i in columns if 'x' in i]
        ys = [i for i in columns if 'y' in i]
        zs = [i for i in columns if 'z' in i]
        xlims = (data.loc[:,xs].min().min(), data.loc[:,xs].max().max())
        ylims = (data.loc[:,ys].min().min(), data.loc[:,ys].max().max())
        zlims = (data.loc[:,zs].min().min(), data.loc[:,zs].max().max())
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        show = matplotlib.animation.FuncAnimation(fig, update, frames = data.index, interval=1)
        return show
            
def anipose_triangulate(anipose_folder):
    os.chdir(os.path.join(anipose_folder, 'temp_project'))
    triangulate = 'anipose triangulate'
    k = subprocess.run(triangulate.split()) 

def anipose_label(anipose_folder, combined = True, filtered_data = True):
    """Automate moving raw videos to temp_project copy, ensuring names match,
    creating 3d and (optionally) combined videos. Obviously very storage/time intensive so 
    not really worth doing on a batch level"""
    os.chdir(anipose_folder)
    trials = os.listdir()
    trials = [i for i in trials if (os.path.isdir(i) and not i.startswith('temp'))]
    for trial in trials:
        real_raw_vid = os.path.join(trial, 'videos-raw')
        temp_raw_vid = os.path.join('temp_project', f'temp_{trial}', 'videos-raw')
        os.makedirs(temp_raw_vid, exist_ok = True)
        transfer_files = os.listdir(real_raw_vid)
        def fix_vid_name(file, filtered_data):
            """Videos from the original project folder are given new names"""
            if not file.endswith('.avi'):
                return
            prefix = file.split('.')[0]
            if filtered_data:
                if not 'filtered' in prefix:
                    prefix = prefix + '_filtered'
            final_name = prefix + '_copyfile.avi'
            return os.path.join(temp_raw_vid, final_name)
        real_files = [os.path.join(real_raw_vid, i) for i in transfer_files if i.endswith('.avi')]
        transfer_files = [fix_vid_name(i, filtered_data) for i in transfer_files if fix_vid_name(i, filtered_data)]
        for real, temp in zip(real_files, transfer_files):
            shutil.copy(real, temp)
    os.chdir(os.path.join(anipose_folder, 'temp_project'))
    label = 'anipose label-3d'
    k = subprocess.run(label.split()) 
    label = 'anipose label-combined'
    k = subprocess.run(label.split())
    
def stitch_from_seconds(trial, minute):
    local_3d = os.path.join(trial, 'pose-3d')
    files = os.listdir(local_3d)
    files = [os.path.join(local_3d, i) for i in files if f'{minute}_second' in i]
    files = sorted(files, key = lambda x: int(re.search('second_(.+).csv', x)[1]))
    assert len(files) == 60
    all_minute = []
    for file in files:
        data = pd.read_csv(os.path.join(local_3d, file))
        all_minute.append(data)
    all_minutes = pd.concat(all_minute)
    all_minutes = all_minutes.reset_index(drop = True)
    return all_minutes

def replace_with_seconds(trial, bad_minutes):
    local_3d = os.path.join(trial, 'pose-3d')
    files = os.listdir(local_3d)
    for minute in bad_minutes:
        og_file = [os.path.join(local_3d, i) for i in files if f'copyfile_{minute}.csv' in i][0]
        os.remove(og_file)
        new_file = stitch_from_seconds(trial, minute)
        new_file.to_csv(og_file, index = False)
    local_3d = os.path.join(trial, 'pose-3d')
    files = os.listdir(local_3d)
    for file in files:
        if '_second_' in file:
            os.remove(os.path.join(local_3d, file))

def return_from_temp_folder(anipose_folder, trial):
    local_3d = os.path.join(trial, 'pose-3d')
    files = [i for i in os.listdir(local_3d) if i.endswith('copyfile.csv')]
    if not len(files) == 1:
        return
    file_3d = files[0]
    file_3d = os.path.join(local_3d, file_3d)
    ind_name = os.path.split(trial)[-1]
    ind_name = re.match('temp_(.+)', ind_name)[1]
    real_path = os.path.join(anipose_folder, ind_name, 'pose-3d')
    if not os.path.isdir(real_path):
        os.makedirs(real_path)
    shutil.copy(file_3d, real_path)

def process(anipose_folder, filtered_data = True, strict_level = 'low', truncate = False, overwrite = False):
    """ Run whole analysis pipeline, creating temp project folder, running triangulation and
    repeatedly checking the results to find and remove time slices with significant reconstruction
    errors
    """
    print(f'params: \n{anipose_folder=} \n{filtered_data=} \n{strict_level=} \n{truncate=}')
    create_project_clone(anipose_folder, filtered_data)
    trials = get_temp_trials(anipose_folder, overwrite) # Skips temp folders with completed pose-3d data
    for trial in trials:
        if truncate:
            print(f'Working on this trial: {trial}\n Using truncated data\n Splitting into minutes...\n\n\n\n')
            project_preprocess(trial, True, truncate)
        else:
            print(f'Working on this trial: {trial}\nSplitting into minutes...\n\n\n\n')
            project_preprocess(trial)
        split_to_minutes(trial)
        anipose_triangulate(anipose_folder)

        print(f'Now looking for triangulation errors in the minute chunks...\n\n\n\n')
        bad_minutes = check_errors(trial, 'minutes', 'absolute', strictness=strict_level)
        if bad_minutes is not None:
            print('Detected bad chunks, splitting bad chunks into seconds now...\n\n\n\n')
            split_to_seconds(trial, bad_minutes)

            anipose_triangulate(anipose_folder)

            print('Now looking for triangulation errors in the second chunks...\n\n\n\n')
            bad_seconds = check_errors(trial, 'seconds', 'absolute', strictness=strict_level)
            if bad_seconds is None:
                print('Failed to find errors that we know exist, switching to another method...\n\n\n\n')
                bad_seconds = check_errors(trial, 'seconds', 'max', strictness=strict_level)
                if bad_seconds is None:
                    print('Failed to find errors again, trying one more method...\n\n\n\n')
                    bad_seconds = check_errors(trial, 'seconds', 'iqr', strictness=strict_level)
                    if bad_seconds is None:
                        print(
                            'We somehow failed to find any seconds containing triangulation errors in this minute\n\n\n\n')
                        bad_seconds = []

            print('Now replacing problematic seconds with nan in corresponding 2D minute chunks\n\n\n\n')
            fix_bad_chunks(trial, bad_seconds)
            clean_up(trial, bad_minutes)

            anipose_triangulate(anipose_folder)

            print('Doing a final check for errors in the complete triangulation...\n\n\n\n')
            bad_minutes = check_errors(trial, 'minutes', 'absolute', strictness=strict_level)
            if bad_minutes is None:
                print('Success, now stitching 3D minute chunks into one final CSV\n\n\n\n')
                fig = stitch_minutes(trial, True)
                print(f'\n\n\nAll in all we lost {len(bad_seconds)} seconds\n\n\n')
            else:
                print('Possible triangulation error or just jumps, check and then delete verify file')
                Path(os.path.join(trial, 'pose-3d', 'verify.txt')).touch()
                fig = stitch_minutes(trial, True)
        else:
            print('Wow, no triangulation errors anywhere. Stitching 3D minute chunks into one final CSV\n\n\n\n')
            stitch_minutes(trial, True)
        return_from_temp_folder(anipose_folder, trial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='Path to a project folder', required=True)
    parser.add_argument('--use_unfiltered', action='store_true', help='Set if you want to use unfiltered DLC tracking data', required=False)
    parser.add_argument('--strict', action='store_true', help='Lower the threshold to detect triangulation errors', required=False)
    parser.add_argument('--truncate', action='store_true', help='Only use for testing params; creates shortened tracking files for faster triangulation', required=False)
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting of previous triangulated data', required=False)
    args = parser.parse_args()

    if args.strict is None:
        strict_level = 'low'
    else:
        strict_level = 'high'

    process(args.project, args.use_unfiltered, strict_level, args.truncate, args.overwrite)
    #anipose_label(anipose_folder, combined = True, filtered_data = filtered_data)
