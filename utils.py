#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:43:21 2023

@author: nathanieltse

this utils.py version is used in the main GUI and should be 
considered the mainline utils file
"""


import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import re
import cv2
import numpy as np
import math
from matplotlib.widgets import Slider
import pickle
import umap
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.animation
from itertools import combinations
from sklearn.preprocessing import normalize
#matplotlib.use('Qt5Agg')
#matplotlib.use('Agg')

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

class svd_3d:
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
    def __init__(self, file, etho_file, manual_z_flip = None, show = False):
        self.etho_file = etho_file
        self.file = file
        if not self.detect_pickle('svd_rotated_df.pickle', 'detect'):
            rotated_df = self.svd(manual_z_flip, show = False)
            self.drop_pickle('svd_rotated_df.pickle', rotated_df)
        else:
            rotated_df = self.detect_pickle('svd_rotated_df.pickle', 'set')
        
        self.rotated_df_ = rotated_df # With underscore indicates raw oriented mouse data, not facing north
        # Nowadays we do the rotation during setup so no need to increase loading times here
        #self.rotated_df = self.rotate_north() # No underscore indicates oriented and north-facing data
        #self.etho = self.align_etho()
        
    def get_data(self, bp_name, frame):
        bp_cords = [bp_name + '_x', bp_name + '_y', bp_name + '_z']
        bp_cords = frame[bp_cords]
        return bp_cords
    
    # Since it takes so long to do the rotate_north function for every frame, we
    # will drop the rotated dataframe as a pickle and re-load it if it exists.
    # The pickle will be dropped and will be searched for in the same folder where
    # The 3D tracking csv comes from
    def detect_pickle(self, pickle_name, job = None):
        directory = os.path.split(self.file)[0]
        if pickle_name in os.listdir(directory):
            if job == 'detect':
                return True
            elif job == 'set':
                with open(os.path.join(directory,pickle_name), 'rb') as handle:
                    temp = pickle.load(handle)
                    print('loaded a pickle')
                    return temp
            else:
                print('no job specified')
        else:
            return False
    
    def drop_pickle(self, pickle_name, pickle_assignment):
        directory = os.path.split(self.file)[0]
        with open(os.path.join(directory,pickle_name), 'wb') as handle:
            pickle.dump(pickle_assignment, handle)
            print('dumped a pickle')

    # Takes x,y,z coords, returns rotated x,y,z coordinates
    def rotate(self, cords, axis, theta):
        if axis == 'z':
            rot = np.array([[math.cos(theta), -math.sin(theta), 0],
                           [math.sin(theta), math.cos(theta), 0],
                           [0, 0, 1]])
        elif axis == 'y':
            rot = np.array([[math.cos(theta), 0, math.sin(theta)],
                              [0, 1, 0],
                              [-math.sin(theta), 0, math.cos(theta)]])
        elif axis == 'x':
            rot = np.array([[1, 0, 0],
                              [0, math.cos(theta), -math.sin(theta)],
                              [0, math.sin(theta), math.cos(theta)]])
        return np.matmul(rot, cords)
    
    def svd(self, manual_z_flip = None, show = False):
        svd_start_time = time.time()
        file = self.file
        file = pd.read_csv(file)
        if not self.detect_pickle('raw_frames_data.pickle', 'detect'):
            # First need to filter out non-coordinate columns
            all_x = pd.concat([file[i + '_x'] for i in svd_3d.bodyparts], axis = 1)
            all_y = pd.concat([file[i + '_y'] for i in svd_3d.bodyparts], axis = 1)
            all_z = pd.concat([file[i + '_z'] for i in svd_3d.bodyparts], axis = 1)
            # Then while we have everything seperated by axis, we get centroid values
            df = pd.concat([all_x.mean(axis=1),
                            all_y.mean(axis=1),
                            all_z.mean(axis=1)],
                           axis = 1)
            # And use those centroid values to normalize so that we can have a zero'd mouse
            norm_df = []
            for i in range(len(svd_3d.bodyparts)):
                temp_df = pd.concat([all_x.iloc[:, i],
                                     all_y.iloc[:, i],
                                     all_z.iloc[:, i]],
                                    axis = 1)
                norm_df.append(temp_df.sub(df.values, axis = 0))
            norm_df = pd.concat(norm_df, axis = 1)

            self.drop_pickle('raw_frames_data.pickle', [df, norm_df])
        else:
            raw_frames_data = self.detect_pickle('raw_frames_data.pickle', 'set')
            df = raw_frames_data[0]
            norm_df = raw_frames_data[1]
        U, S, Vh = np.linalg.svd(df.values, full_matrices=True)
        er = np.matmul(df, Vh.T)
    
        rotated_raw = []
        for i in range(0, norm_df.shape[1], 3):
            bp_arr = norm_df.iloc[:, i:i+3]
            bp_rotated= np.matmul(bp_arr, Vh.T)
            rotated_raw.append(bp_rotated)
        rotated = pd.concat(rotated_raw, axis = 1)
        rotated.columns = norm_df.columns
            
        """SVD finds the plane of best fit given the mouse's average location over a whole video, 
        which allows us to figure out essentially what rotation we need to apply to produce 
        a flat ground. However, this only accounts for the x,y axis, not the Z. Sometimes
        the SVD will result in the mouse being flipped (i.e. the ground is zero but the ceiling is
        -inf). We need a way to figure out whether this happened"""
        
        ear_up = rotated['l_ear_mousper_z'] - rotated['tailbase_z'] # Replaces that horrible z_flip assessment in previous versions
        if ear_up.mean() < 0:
            rotated[all_z.columns] = rotated[all_z.columns] * -1

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(121, projection ='3d')
            ax.scatter(file.iloc[:,0], file.iloc[:,1], file.iloc[:,2], c = file.index, cmap = 'viridis')
            ax = fig.add_subplot(122, projection ='3d')
            ax.scatter(rotated.iloc[:,0], rotated.iloc[:,1], rotated.iloc[:,2] * -1, c = rotated.index, cmap = 'viridis')
            fig_name = re.search('(.+)-frame', self.file.split('/')[-1])[1]
            plt.suptitle(fig_name)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            for row, value in rotated.iloc[:100:25].iterrows():
                self.show_frame(value, ax, True)
            plt.suptitle(fig_name)
        print(f'SVD step took {time.time() - svd_start_time} seconds')
        return rotated
    
    """We can finally put everything together. For a given frame, we first orient
    the 3D data using the theta values we got from the above code blocks so that we
    are looking at a top-down (ish) view of the mouse. We then find the angle we need to 
    rotate in order to point the orienting bodypart directly north (x = 0, y = magnitude).
    We can then plot out the newly oriented data in 2d. Also an option to see what the data
    looked like after pre-rotating but before north-orienting"""
    def rotate_north(self, show_rotated = False, show_original = False, old_way = False, orient_bp = 'nose'):
        frame = self.rotated_df_
        rotated_north_frames = []
        counter = 0
        #total_time = time.time()
        #start_time = time.time()
        for row_num, frame_ in frame.iterrows():
            if counter / len(frame) > 0.05:
                #print(f'prop. done: {len(rotated_north_frames) / len(frame)}')
                #print(f'took {time.time() - start_time} seconds')
                #start_time = time.time()
                counter = 0
            # Getting the x,y,z coords for the bodypart to orient around
            orient_bp = orient_bp
            orient_cords = [orient_bp + '_x', orient_bp + '_y', orient_bp + '_z']
            orient = frame_[orient_cords]
            if show_original:
                fig = plt.figure()
                ax1 = fig.add_subplot(2,1,1,adjustable='box', aspect='equal')
                ax1.scatter(orient[0], orient[1], marker = '*')
            # Get euclidian distance from 0,0
            mag = np.linalg.norm([orient[0], orient[1]])                                   # Equivalent to math.sqrt((orient[0] ** 2) + (orient[1] ** 2))
            if show_original:
                ax1.scatter(0, mag, marker = 'x')
            # Get theta between current nose position and position that maxes y value (0, mag)
            theta = np.arccos(
                np.dot([orient[0], orient[1]], [0, mag])
                / (np.linalg.norm([orient[0], orient[1]]) * np.linalg.norm([0, mag]))
                )
            # Theta is always less than pi, so we need to adjust which direction we are rotating
            # depending on if the orienting point has an x value that is negative or positive
            if orient[0] < 0:
                theta = -theta
            # Provide the z-axis rotation matrix right away
            rot = np.array([[math.cos(theta), -math.sin(theta), 0],
                           [math.sin(theta), math.cos(theta), 0],
                           [0, 0, 1]])
            if old_way:
                #fig = plt.figure()
                rotated_array = pd.DataFrame()
                if not show_original and not show_rotated:
                    for bodypart in svd_3d.bodyparts:
                        x,y,z = self.get_data(bodypart, frame_)
                        x,y,z = self.rotate([x,y,z], 'z', theta)
                        rotated_array[bodypart] = {'x' : x,
                                                   'y' : y,
                                                   'z' : z}
                    rotated_frame = self.return_to_normal(rotated_array)
                    
                else:
                    for limbsys in [svd_3d.backbone, svd_3d.front_limbs, svd_3d.back_limbs, svd_3d.ears]:
                        parts = []
                        for bodypart in limbsys:
                            x,y,z = self.get_data(bodypart, frame_)
                            x,y,z = self.rotate([x,y,z], 'z', theta)
                            rotated_array[bodypart] = {'x' : x,
                                                       'y' : y,
                                                       'z' : z}
                            if show_rotated:
                                ax1.scatter(x,y)
                                parts.append([x,y])
                        if show_rotated:
                            ax1.plot([i[0] for i in parts],
                                    [i[1] for i in parts])
                    rotated_frame = self.return_to_normal(rotated_array)
                    
                    if show_original:
                        ax2 = fig.add_subplot(2,1,2,adjustable='box', aspect='equal')
                        for limbsys in [svd_3d.backbone, svd_3d.front_limbs, svd_3d.back_limbs, svd_3d.ears]:
                            parts = []
                            for bodypart in limbsys:
                                x,y,z = self.get_data(bodypart, frame_)
                                #x,y,z = rotate([x,y,z], 'z', theta)
                                ax2.scatter(x,y)
                                parts.append([x,y])
                            ax2.plot([i[0] for i in parts],
                                    [i[1] for i in parts])
                        return
            else:
                rotated_array = pd.DataFrame()
                f = frame_.values.reshape(-1,3) # Automatically makes the frame series object into an array with bodyparts as rows and x,y,z as columns
                f = np.matmul(f, rot.T)
                rotated_frame = f.reshape(51,)
            rotated_north_frames.append(rotated_frame)
            counter += 1
        rotated_north_frames = pd.DataFrame(rotated_north_frames)
        rotated_north_frames.columns = frame.columns
        #print(f'total time took: {time.time() - total_time}')
        return rotated_north_frames
    
    # Get only body part tracking data from an entire frame; necessary if using 
    # pre-processing scaling as things like likelihood and n_cams will affect data spread
    def filter_frame_data(self, frame):
        all_data = pd.DataFrame()
        for bodypart in svd_3d.bodyparts:
            names = [bodypart + '_x', bodypart + '_y', bodypart + '_z']
            x,y,z = frame[names]
            all_data[bodypart] = {'x' : x,
                                  'y' : y,
                                  'z' : z}
        return all_data

    # Normalize and preserve distances. Takes in frame data rearranged as x,y,z per bodypart
    # e.g. the output of filter_frame_data()
    def norm_all_dims(self, frame_array):
        x_max = np.max(frame_array.loc['x'])
        y_max = np.max(frame_array.loc['y'])
        z_max = np.max(frame_array.loc['z'])
        
        x_min = np.min(frame_array.loc['x'])
        y_min = np.min(frame_array.loc['y'])
        z_min = np.min(frame_array.loc['z'])
        
        centroid = np.array([np.mean(frame_array.loc['x']),
                    np.mean(frame_array.loc['y']),
                    np.mean(frame_array.loc['z'])])
        moved = frame_array.T - centroid
        return moved
        
    # Take in output of norm_all_dims and return to raw frame data format
    # Requires 'x','y','z' as column names, bodyparts as index, but will automatically
    # transpose the dataframe to fit the naming convention
    def return_to_normal(self, frame_array):
        if len(frame_array) == 3:
            frame_array = frame_array.T
        names = []
        for i in frame_array.index:
            for n in frame_array.columns:
                name = i + '_' + n
                names.append(name)
        return pd.Series(frame_array.values.reshape(-1), index = names)    
    def align_etho(self):
        if self.etho_file is None:
            frame_ethos = pd.Series(index = range(len(self.rotated_df)))
            frame_ethos[0:len(self.rotated_df)] = -1
            return frame_ethos
        etho = pd.read_csv(self.etho_file, header = 15)
        if 'Behavior' not in etho.columns:                                     # Such is the case if you used two videos or more to score 
            for i in range(1, 4):
                etho = pd.read_csv(self.etho_file, header = 15 + i)
                if 'Behavior' in etho.columns:
                    break
        if not all(etho['Status'][::2] == 'START') and all(etho['Status'][1::2] == 'STOP'):
            raise IndexError('Etho file error: START/STOP does not alternate properly')
        if not all(x == z for x,z in zip(etho['Media file path'][::2], etho['Media file path'][1::2])):
            raise IndexError('Etho file error: Only one type of behavior can be STARTed at a time')
        etho_data = []
        last_row = None
        for count, row in etho.iterrows():
            if last_row is None:
                etho_data.append(int(np.round(row['Time'] * 30)) * [0])
                last_row = row
            elif row['Status'] == 'START':
                etho_data.append(int(np.round((row['Time'] - last_row['Time']) * 30)) * [0])
                last_row = row
            elif row['Status'] == 'STOP':
                etho_data.append(int(np.round((row['Time'] - last_row['Time']) * 30)) * [row['Behavior']])
                last_row = row
        etho_data = [value for subvalue in etho_data for value in subvalue]
        frame_ethos = pd.Series(etho_data)
        len_dif = len(self.rotated_df) - len(frame_ethos)
        if len_dif < 0:
            frame_ethos = frame_ethos.iloc[:len(self.rotated_df)]
        elif len_dif > 0:
            temp = pd.Series([0] * len_dif)
            frame_ethos = pd.concat([frame_ethos, temp]).reset_index(drop=True)
        return frame_ethos      
    
    def show_frame(self, frame, ax, threed = False):  
        if threed:
            all_syss = []
            for limbsys in [svd_3d.backbone, svd_3d.front_limbs, svd_3d.back_limbs, svd_3d.ears]:
                parts = []
                for bodypart in limbsys:
                    x,y,z = self.get_data(bodypart, frame)
                    ax.scatter(x,y,z, alpha = .5)
                    parts.append([x,y,z])
                ax.plot([i[0] for i in parts],
                        [i[1] for i in parts],
                        [i[2] for i in parts],
                        alpha = .5)
                all_syss.append(parts)
            
        else:
            all_syss = []
            for limbsys in [svd_3d.backbone, svd_3d.front_limbs, svd_3d.back_limbs, svd_3d.ears]:
                parts = []
                for bodypart in limbsys:
                    x,y,z = self.get_data(bodypart, frame)
                    parts.append([x,y,z])
                all_syss.append(parts)
            for system in all_syss:
                plt.scatter([i[0] for i in system], [i[1] for i in system])
                plt.plot([i[0] for i in system], [i[1] for i in system])
    def visualize(self, video, ss = 1, slider = False, start_frame = 0):
        frame_cords = self.rotated_df
        df = self.rotated_df
        if 'velocity' in df.columns:
            df = df.loc[:, ~df.columns.isin(['velocity'])]
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df.T.values).T,
                            index = df.index,
                            columns = df.columns)
        if 'velocity' in self.rotated_df.columns:
            df['velocity'] = self.rotated_df.loc[:, 'velocity']
        has_nan = np.isnan(df).any(axis=1)
        df = df.loc[~has_nan]
        etho = self.etho.loc[~has_nan]

        #wanted_bp = [i for i in frame_cords.columns if 'center' not in i]
        #wanted_bp = [i for i in wanted_bp if 'tailtip' not in i]
        #frame_cords = frame_cords[wanted_bp]

        n_neighbors = 14
        min_dist = 0.6
        n_components = 2
        reducer = umap.UMAP(n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            n_components=n_components)
        embedding = reducer.fit_transform(df.values)
        embed_df = pd.DataFrame(embedding, index = df.index)
        cap = cv2.VideoCapture(video)
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(133)
        ax = fig.add_subplot(212, projection='3d')
        ax.view_init(elev = 0, azim = 173)
        ax.axis('off')
        face_points = embed_df[etho == 'face groom']
        body_points = embed_df[etho == 'body groom']
        rear_points = embed_df[etho == 'rear']
        #ax2.scatter(embed_df[0], embed_df[1], c = np.arange(len(embed_df)), cmap = 'viridis', alpha=.3)
        ax2.scatter(embed_df[0], embed_df[1], color = 'black', alpha = .05, s = ss)
        ax2.scatter(face_points[0], face_points[1], color = 'red', alpha = 0.05, s = ss)
        ax2.scatter(body_points[0], body_points[1], color = 'pink', alpha = 0.05, s = ss)
        ax2.scatter(rear_points[0], rear_points[1], color = 'blue', alpha = 0.05, s = ss)
        columns = [i for i in df.columns if '_' in i]
        columns = [i for i in columns if i.split('_')[-1] in ['x','y','z']]
        xs = [i for i in columns if 'x' in i]
        ys = [i for i in columns if 'y' in i]
        zs = [i for i in columns if 'z' in i]
        xlims = (df.loc[:,xs].min().min(), df.loc[:,xs].max().max())
        ylims = (df.loc[:,ys].min().min(), df.loc[:,ys].max().max())
        zlims = (df.loc[:,zs].min().min(), df.loc[:,zs].max().max())
        def update_slider(val):
            frame = int(frame_slider.val)
            update(frame)
        def update(i):
            ax.clear()
            ax2.clear()
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            self.show_frame(df.loc[i], ax, threed=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the video frame
            rere, frfr = cap.read()  # Read the frame
            if rere:
                ax1.clear()
                ax1.imshow(frfr)  # Display the video frame
            ax2.set_xlim(embed_df[0].min(), embed_df[0].max())
            ax2.set_ylim(embed_df[1].min(), embed_df[1].max())
            ax2.scatter(embed_df[0], embed_df[1], color = 'black', alpha = .05, s = ss)
            ax2.scatter(face_points[0], face_points[1], color = 'pink', alpha = 0.5, s = ss)
            ax2.scatter(body_points[0], body_points[1], color = 'red', alpha = 0.5, s = ss)
            #ax2.scatter(rear_points[0], rear_points[1], color = 'blue', alpha = 0.5, s = ss)
            #show_frame(frames[i], threed=True)
            ax2.scatter(embed_df.loc[i][0], embed_df.loc[i][1], color = 'yellow', marker = '*')
            ax.axis('off')
        if slider == True:
            ax3 = fig.add_subplot(313)
            frame_slider = Slider(ax3, 'Frame', 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, valinit=0)
            frame_slider.on_changed(update_slider)
        else:
            if start_frame != 0:
                self.ani = matplotlib.animation.FuncAnimation(fig, update, frames = range(start_frame, len(embed_df)), interval=1)
            else:
                self.ani = matplotlib.animation.FuncAnimation(fig, update, frames = embed_df.index, interval=1)
    def av_velocity(self):
        file = self.file
        file = pd.read_csv(file)
        if not self.detect_pickle('velocity_data.pickle', 'detect'):
            vel_start_time = time.time()
            file_frames = []
            raw_frames = []
            for count, value in file.iterrows():
                pre_frame = self.filter_frame_data(value)
                file_frames.append(pre_frame.T.mean())
            df = pd.DataFrame(file_frames)
            dff = df - df.shift(1)
            dfff = dff.apply(np.linalg.norm, axis = 1)
            self.drop_pickle('velocity_data.pickle', dfff)
            print(f'velocity calculations took {time.time() - vel_start_time} seconds to process')
        else:
            dfff = self.detect_pickle('velocity_data.pickle', 'set')
        self.rotated_df['velocity'] = dfff
    def better_align(self, orient_to = 'center', orient_with = 'tailbase'):
        df = self.rotated_df_
        for col in df.columns:
            axis = col.split('_')[-1]
            df[col] = df[col] - df[f'{orient_with}_{axis}']
        #l_leg, r_leg = df.columns.str.contains('LBack_root'), df.columns.str.contains('RBack_root')
        #between_legs = np.nanmean([df[df.columns[l_leg]], df[df.columns[r_leg]]], axis = 0)
        rotated_north_frames = []
        for i in range(len(df)):
            orient = df.loc[i, df.columns.str.contains(orient_to)].values
            mag = np.linalg.norm([orient[0], orient[1]])                                   # Equivalent to math.sqrt((orient[0] ** 2) + (orient[1] ** 2))
            # Get theta between current nose position and position that maxes y value (0, mag)
            theta = np.arccos(
                np.dot([orient[0], orient[1]], [0, mag])
                / (np.linalg.norm([orient[0], orient[1]]) * np.linalg.norm([0, mag]))
                )
            # Theta is always less than pi, so we need to adjust which direction we are rotating
            # depending on if the orienting point has an x value that is negative or positive
            if orient[0] < 0:
                theta = -theta
            # Provide the z-axis rotation matrix right away
            rot = np.array([[math.cos(theta), -math.sin(theta), 0],
                           [math.sin(theta), math.cos(theta), 0],
                           [0, 0, 1]])
            rotated_array = pd.DataFrame()
            f = df.iloc[i].values.reshape(-1,3) # Automatically makes the frame series object into an array with bodyparts as rows and x,y,z as columns
            f = np.matmul(f, rot.T)
            rotated_frame = f.reshape(51,)
            rotated_north_frames.append(rotated_frame)
        rotated_north_frames = pd.DataFrame(rotated_north_frames)
        rotated_north_frames.columns = df.columns
        #print(f'total time took: {time.time() - total_time}')
        return rotated_north_frames
        

def find_all_pairs(mouse, do_all = False, frame_num = 0):
    df = []
    if do_all:
        cols = []
        for pair in combinations(bodyparts, 2):
            if ('tailtip' in ' '.join(pair)):
                continue
            cords_1 = [pair[0] + i for i in ['_x', '_y', '_z']]
            cords_2 = [pair[1] + i for i in ['_x', '_y', '_z']]
            bp1, bp2 = [mouse.rotated_df.loc[:, i] for i in [cords_1, cords_2]]
            diffs = bp1.values - bp2.values
            df.append(pd.Series(np.linalg.norm(diffs, axis = 1)))
            cols.append(' '.join(pair))
        df = pd.concat(df, axis = 1)
        df.columns = cols
        return df
    for pair in combinations(bodyparts, 2):
        if ('tailtip' in ' '.join(pair)):
            continue
        cords_1 = [pair[0] + i for i in ['_x', '_y', '_z']]
        cords_2 = [pair[1] + i for i in ['_x', '_y', '_z']]
        bp_1_data = mouse.rotated_df.loc[frame_num, cords_1]
        bp_2_data = mouse.rotated_df.loc[frame_num, cords_2]
        df.append(np.linalg.norm(bp_1_data.values - bp_2_data.values))
    df = pd.Series(df)
    return df

def angle_between_bps(bp1, bp2, bp3, mouse, do_all = False, frame_num = 0):
    """Find the angle between three body parts, for either a single frame or all frames.
    Find using the acos method we used during rotate_north"""
    bp1_cords = [bp1 + i for i in ['_x', '_y', '_z']]
    bp2_cords = [bp2 + i for i in ['_x', '_y', '_z']]
    bp3_cords = [bp3 + i for i in ['_x', '_y', '_z']]
    
    if do_all:
        bp1, bp2, bp3 = [mouse.rotated_df.loc[:, i] for i in [bp1_cords, bp2_cords, bp3_cords]]
        vector = lambda part_a, part_b: [part_b.iloc[:, 0] - part_a.iloc[:, 0], 
                                         part_b.iloc[:, 1] - part_a.iloc[:, 1], 
                                         part_b.iloc[:, 2] - part_a.iloc[:, 2]]
        v1 = np.array(vector(bp2, bp1))
        v2 = np.array(vector(bp2, bp3))
        v1,v2 = [normalize(i.T) for i in [v1,v2]]
        v_dot = np.einsum('ij,ij->i', v1, v2)                                  # Honestly I don't understand Einstein notation but this saves significant time 
        angle = np.arccos(v_dot)                                               # compared to doing dot products in a loop so I'm using it anyways
        return angle
    else:                                                                      
        bp1, bp2, bp3 = [mouse.rotated_df.loc[frame_num, i] for i in [bp1_cords, bp2_cords, bp3_cords]]
        vector = lambda part_a, part_b: [part_b[0] - part_a[0], 
                                         part_b[1] - part_a[1], 
                                         part_b[2] - part_a[2]]
        v1 = vector(bp2, bp1)                                                  # https://math.stackexchange.com/questions/361412/finding-the-angle-between-three-points
        v2 = vector(bp2, bp3)                                                  # See the corrected answer in link above for how we choose start/end points for the vector
        v1,v2 = [normalize(np.array([i])).reshape(3,) for i in [v1,v2]]
        v_dot = np.dot(v1, v2.T)
        angle = np.arccos(v_dot)
        return angle
    
def assemble_rotation_data(mouse, select_trips = True):
    mouse_angles = pd.DataFrame()
    if select_trips:
        for trips in [['nose', 'neck', 'half_neck'],
                      ['neck', 'half_neck', 'center'],
                      ['half_neck', 'center', 'half_tailbase'],
                      ['center', 'half_tailbase', 'tailbase'],
                      ['half_tailbase', 'tailbase', 'tailtip'],
                      ['LBack_paw', 'LBack_root', 'tailbase'],
                      ['RBack_paw', 'RBack_root', 'tailbase'],
                      ['LBack_root', 'tailbase', 'half_tailbase'],
                      ['RBack_root', 'tailbase', 'half_tailbase'],
                      ['nose', 'center', 'tailbase']]:
            angle = angle_between_bps(*trips, mouse, True)
            mouse_angles[' '.join(trips)] = angle
    else:
        mouse_angles = pd.concat([pd.Series(angle_between_bps(*i, mouse, True), name = ' '.join(i)) for i in combinations(bodyparts, 3)], axis = 1)
    mouse.angle_df = mouse_angles
    mouse.combined_df = pd.concat([mouse.distance_df, mouse.angle_df], axis = 1)

def consider(data):
    temp = data.value_counts()
    #print(temp)
    main_data = temp.index[0]
    main_weight = temp.iloc[0]
    return main_data, main_weight

def roll_up(mouse, period, dataframe='distance', save_space=True, restrict_bps = None):
    """

    Parameters
    ----------
    mouse : TYPE
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.
    dataframe : TYPE, optional
        DESCRIPTION. The default is 'distance'.
    save_space : TYPE, optional
        DESCRIPTION. The default is True.
    restrict_bps : list, optional
        DESCRIPTION. removes columns with these bps before rolling

    Returns
    -------
    rolled_data : TYPE
        DESCRIPTION.
    rolled_etho : TYPE
        DESCRIPTION.

    """
    count = 0
    rolled_data = pd.DataFrame()
    rolled_etho = pd.DataFrame()
    dataframes = {'distance' : mouse.distance_df,
                  'angle' : mouse.angle_df,
                  'combined' : mouse.combined_df}
    if dataframe in dataframes.keys():
        dff = dataframes[dataframe]
    else:
        dff = mouse.rotated_df.copy()
        if mouse in right_bean:
            print('right bean detected')
            cols = [i for i in dff.columns if '_x' in i]
            dff.loc[:, cols] = dff.loc[:, cols] * -1
        if 'velocity' in dff.columns:
            dff = dff.loc[:, ~dff.columns.isin(['velocity'])]
        scaler = MinMaxScaler()
        dff = pd.DataFrame(scaler.fit_transform(dff.T.values).T,
                           index=dff.index,
                           columns=dff.columns)
        if 'velocity' in mouse.rotated_df.columns:
            dff['velocity'] = mouse.rotated_df.loc[:, 'velocity']
            dff = dff.iloc[1:].reset_index(drop=True)
            print('WARNING: rotated_df and etho data is off by 1 due to dropping velocity nan row')
    rolled_data = []
    rolled_etho = []
    rolled_ids = []                                                        # Contains indicies so we can later unroll embeddings of interest
    if restrict_bps is not None:
        dff = dff[[i for i in dff.columns if not any([e in i for e in restrict_bps])]]
    if save_space:
        while True:
            start = 0 + (period * count)
            stop = period + (period * count)
            if stop >= len(dff):
                break
            rolled_data.append(dff.loc[start:stop].values.reshape(-1,))
            rolled_etho.append(consider(mouse.etho.loc[start:stop]))
            rolled_ids.append((start,stop))
            count += 1
    else:
        while True:
            start = count - (period / 2)
            stop = count + (period / 2)
            #start = count
            #stop = count + period
            if start < 0:
                count += 1
                continue
            if stop >= len(dff):
                break
            rolled_data.append(pd.Series(dff.loc[start:stop].values.reshape(-1,)))
            rolled_etho.append(consider(mouse.etho.loc[start:stop]))
            rolled_ids.append((start,stop))
            count += 1
    rolled_data = pd.concat(rolled_data, axis = 1).T
    rolled_etho = pd.DataFrame(rolled_etho, columns = ['etho', 'weight'])
    rolled_ids = pd.DataFrame(rolled_ids, columns = ['ind_start', 'ind_stop'])
    rolled_etho = pd.concat([rolled_etho, rolled_ids], axis = 1)
    rolled_etho['single_id'] = mouse.single_id
    return rolled_data, rolled_etho

def extract_datatype(mouse, dataframe='distance', roll = False, period = 10, save_space=True):
    if roll:
        data, etho = roll_up(mouse, period, dataframe, save_space)
        inds = etho.loc[etho['etho'].apply(type) == str]
        dff = data.loc[inds.index]
        return inds, dff
    inds = mouse.etho.loc[mouse.etho.apply(type) == str]
    dataframes = {'distance' : mouse.distance_df,
                  'angle' : mouse.angle_df,
                  'combined' : mouse.combined_df}
    if dataframe in dataframes.keys():
        dff = dataframes[dataframe].loc[inds.index]
    #if dataframe == 'distance':
    #    dff = mouse.distance_df.loc[inds.index]
    else:
        dff = mouse.rotated_df.loc[inds.index].copy()
        if mouse in right_bean:
            print('right bean detected')
            cols = [i for i in dff.columns if '_x' in i]
            dff.loc[:, cols] = dff.loc[:, cols] * -1
        if 'velocity' in dff.columns:
            dff_copy = dff.copy()
            dff = dff.loc[:, ~dff.columns.isin(['velocity'])]
        scaler = MinMaxScaler()
        dff = pd.DataFrame(scaler.fit_transform(dff.T.values).T,
                           index=dff.index,
                           columns=dff.columns)
        if 'velocity' in mouse.rotated_df.columns:
            dff['velocity'] = dff_copy.loc[:, 'velocity']
            dff = dff.iloc[1:].reset_index(drop=True)
            print('WARNING: rotated_df and etho data is off by 1 due to dropping velocity nan row')
    return inds, dff

def scan_folder(folder):
    """ folder: absolute path to folder (individual mouse inside a project folder)
    
        Looks for calibration, 2d and 3d folders to verify its an anipose project"""
        
    search_items = ['calibration', 'pose-2d', 'pose-3d']
    contents = os.listdir(folder)
    if all(i in contents for i in search_items):
        return True
    else:
        return False
    
def folder_search(root_folder, projects = 'all', nesting = 1):
    """
    root_folder: absolute path to folder containing anipose projects
    
    projects: list containing names of project folders you want to use. If 
             default 'all' is used, all folders in the root_folder will be
             scanned.
                
    Goes through folders of interest, sees which folders contain 3D mouse data.
    Scans folders inside a project folder by looking for calibration, pose-2d,
    and pose-3d folders.
    """
   
    valid_mice = []
    if projects == 'all':
        projects = os.listdir(root_folder)
        projects = [i for i in projects if not i.startswith('.') and os.path.isdir(os.path.join(root_folder, i))]
    for project in projects:
        abs_path = os.path.join(root_folder, project)
        if nesting < 1:
            if scan_folder(abs_path):
                valid_mice.append(abs_path)
            continue
        else:
            mouse_folders = [i for i in os.listdir(abs_path) if os.path.isdir(os.path.join(abs_path, i))]
            for mouse_folder in mouse_folders:
                abs_folder = os.path.join(abs_path, mouse_folder)
                if scan_folder(abs_folder):
                    valid_mice.append(os.path.join(abs_path, mouse_folder))
    return valid_mice

def setup_mouse(path):
    """
    path: absolute path to a project subfolder (mouse level)
    
    Will return the 3D tracking file for that subfolder
    """
    filename = os.path.split(path)[-1]
    
    temp = os.listdir(os.path.join(path, 'pose-3d'))
    tracking_file = [i for i in temp if i.endswith('copyfile.csv')]        # All stitched 3d files should end with this rather than 'copyfile_9.csv'
    assert len(tracking_file) == 1, ('No single stitched file detected - likely failed to fix\
                                     bad minutes. Try using a lower strictness')
                        
    temp = os.listdir(path)
    etho_file = [i for i in temp if i.endswith('.csv')]
    etho_file = [i for i in etho_file if not i.startswith('._')]
    if len(etho_file) == 0:
        etho_file = None
    elif len(etho_file) > 1:
        display = zip(range(len(etho_file)), etho_file)
        print(f'Error: multiple possible etho files. Choose from this list:')
        [print(c,v) for c,v in display];
        selected = input('>')
        etho_file = etho_file[int(selected)]
    else:
        print('One etho file found')
        etho_file = etho_file[0]
    if etho_file is not None:
        etho_file = os.path.join(path, etho_file)
    return os.path.join(path, 'pose-3d', tracking_file[0]), etho_file

def sort_tests(path, input_object, output_dict):
    """

    Parameters
    ----------
    path : str
        The path of the object, or a file name.
    input_object : anything
        The thing you want sorted into the dictonary.
    output_dict : dict
        The dict where you want to sort into.

    Returns
    -------
    dict
    
    Sort svd_3d objects or just files into any pre-created dictionary that has 
    the structure: 
        example = {'wt' : {},
                'wt-ca' : {},
                'card9' : {},
                'card9-ca' : {},
                'other' : {}}

    """
    folder_name = os.path.split(path)[1]
    if 'card9' in folder_name:
        if '-ca-' in folder_name:
            key = 'card9-ca'
        else:
            key = 'card9'
    elif 'wt' in folder_name:
        if '-ca-' in folder_name:
            key = 'wt-ca'
        else:
            key = 'wt'
    else:
        key = 'other'
    if type(input_object) == svd_3d:
        input_object.group_key = key
    filename = os.path.split(input_object.file)[-1]
    input_object.date = re.search('(.+-.+-..)_', filename)[1]
    input_object.single_id = re.search('(.+)-frame_synced', filename)[1]
    output_dict[key][input_object.single_id] = input_object
    return output_dict
            
def prep_for_train(mouse):
    """ 
    mouse: svd_3d object
    
    Prepares information that will be used by etho_roll and data_roll
    """
    filename = mouse.file.split('/')[-1]
    try:
        mouse.date = re.search('(.+-.+-..)_', filename)[1]
        mouse.single_id = re.search('(.+)-frame_synced', filename)[1]
        print(mouse.date, mouse.single_id)
    except TypeError:
        mouse.single_id = re.search('(.+)-frame_synced', filename)[1]
    if mouse.detect_pickle('final_rotated_df.pickle', 'detect'):
        mouse.rotated_df = mouse.detect_pickle('final_rotated_df.pickle', 'set')
    else:
        mouse.rotated_df = mouse.better_align()    
        mouse.drop_pickle('final_rotated_df.pickle', mouse.rotated_df)
    if mouse.detect_pickle('final_distance_df.pickle', 'detect'):
        mouse.distance_df = mouse.detect_pickle('final_distance_df.pickle', 'set')
    else:
        mouse.distance_df = find_all_pairs(mouse, do_all = True)
        mouse.drop_pickle('final_distance_df.pickle', mouse.distance_df)
    assemble_rotation_data(mouse, False)    
    mouse.etho = mouse.align_etho()
    """mouse.tailbase_df = mouse.rotate_north(show_rotated = False, 
                                           show_original = False, 
                                           old_way = False, 
                                           orient_bp = 'tailbase')"""
        
def arrange_data(name_list, clustered_df, etho_roll, behs, only_good_injs = False):
    """
    NOT the same as the one found in umap_model_train (or previous versions as
    scratch-named files). This one doesn't require svd_3d objects, just file names.
    
    
    behs is highly modifiable - just keep adding behaviors as long as you have
    a list of cluster IDs to go along with it, or add more IDs to a pre-existing 
    behavior (but keep it a list). Examples:
    
    
    behs = {'face grooming' : [2], 
            'body grooming' : [1, 0],
            'rearing' : [3]}
    
    behs = {'face grooming' : [0], 
            'right side grooming' : [1],
            'left side grooming' : [3],
            'midline grooming' : [2],
            'hind leg grooming' : [4],
            'rearing' : [6],
            'wall rearing' : [5]
            }
    """
    
    """
    Create treatment_df to store outputs and also to provide a searchable index
    for determining which data is pre-CNO experiments
    """
    treatment_index = pd.read_csv('/Users/nathanieltse/Library/CloudStorage/Box-Box/card9_coding/data/identity.csv')
    injection_df = treatment_index['Injection Accuracy']
    treatment_index = [(row['group'], row['treatment'], '-'.join(row['date'].split('/')), row['ID']) for count,row in treatment_index.iterrows()]
    treatment_index = pd.MultiIndex.from_tuples(treatment_index, names = ['group', 'treatment', 
                                                                          'date', 'ID'])
    treatment_df = pd.DataFrame(index = treatment_index, columns = behs.keys())
    injection_df.index = treatment_index
    
    
    def extract_data(name, behs):
        inds = np.where(etho_roll['single_id'] == name)[0]
        if len(inds) == 0:
            raise NameError(f'Name {name} does not show up in etho_roll')
        c_df = clustered_df['cluster_label'].iloc[inds]
        extracted = {}
        for key, cluster_ids in behs.items():
            cluster_ids = [i for i in cluster_ids if i in c_df.value_counts().index]
            extracted[key] = sum([c_df.value_counts()[i] for i in cluster_ids])
        return extracted
        
    def sort_non_cno(names, behs):
        """
        Input a list of names, get back a MultiIndexed DataFrame

        """
        non_cno_index = pd.MultiIndex.from_product([['wt', 'card9', 'candida', 'card9-ca'],
                                                    behs.keys()], 
                                                   names = ['group', 
                                                            'behavior'])
        non_cno_df = pd.DataFrame(index = non_cno_index)
        for name in names:
            print(f'non-cno: {name}')
            if 'wt-ca' in name:
                key = 'candida'
            elif all([i in name for i in ['card9', '-ca-']]):
                key = 'card9-ca'
            elif 'wt' in name:
                key = 'wt'
            elif 'card9' in name:
                key = 'card9'
            else:
                print(f'Non-CNO: this name could not be assigned a group: {name}')
            extracted = extract_data(name, behs)
            non_cno_df[name] = pd.Series()
            non_cno_df[name][key] = pd.Series(extracted)
        return non_cno_df
        
    def sort_cno(names, behs):
        """ 
        Search across the treatment_df index until some components match and deliver an identity:
            e.g. [('CARD9-/- w.o c.a.', 'Saline', '7-18-24', 'F13')]
        Then use that to decide where the data should be placed
        """
        for name in names:
            #print(f'working on cno: {name}')
            date = re.search('(.+-.+-.+?)_', name)[1]
            ref_num = re.search('_([FM][0-9]+)', name)[1] # Match any substring that starts with a F or M after an underscore, up until the next underscore
            try:
                treatment = re.search('saline|CNO', name)[0]
                components = [date, ref_num, treatment]
            except TypeError:
                print(f'{name} >> baseline')
                components = [date, ref_num]
            identities = [i for i in treatment_df.index if all([component in i for component in components])]
            assert len(identities) != 0, f'CNO: this name could not be assigned: {name}'
            assert not len(identities) > 1, f'CNO: this returned too many assignments: {name}'
            g,t,d,i = identities[0]
            extracted = extract_data(name, behs)
            if t in name:
                treatment_df.loc[(g, t, d, i), :] = pd.Series(extracted) 
            else:
                treatment_df.loc[(g, 'baseline', d, i), :] = pd.Series(extracted)
        unused = treatment_df.loc[treatment_df.isna().any(axis=1)].index
        print(f'No data provided for these individuals: {unused}')
        return treatment_df # Index is created from identities.csv so will include unincluded names
    
    def sort_card9cre(names, behs):
        if len(names) == 0:
            return None
        sort_df = pd.read_csv('/Users/nathanieltse/Library/CloudStorage/Box-Box/card9_coding/data/card9cre_identity.csv')
        card9cre_index = pd.MultiIndex.from_product([sort_df['Genotype'].unique(),
                                                    behs.keys()], 
                                                   names = ['group', 
                                                            'behavior'])
        card9cre_df = pd.DataFrame(index = card9cre_index)
        for name in names:
            ref_num = int(re.search('CARD9Cre_(.+)', name)[1])
            genotype = sort_df.loc[sort_df['Ear tagging'] == ref_num, 'Genotype']
            print(f'{name} >>> {genotype.values[0]}')
            extracted = extract_data(name, behs)
            card9cre_df[name] = pd.Series()
            card9cre_df[name][genotype.values[0]] = pd.Series(extracted)
        return card9cre_df
    
    def only_good_injections(cno_df, strictness = 'soft'):
        final_df = []
        if strictness == 'hard': 
            approval = ['y']
        else: 
            approval = ['y', '?']
        for index, row in cno_df.iterrows():
            group, treatment, date, ID = index
            injection_quality = injection_df.loc[(slice(None), slice(None), date, ID)]
            if injection_quality.item() in approval:
                final_df.append(row)
        final_df = pd.concat(final_df, axis = 1).T
        return final_df
        
        
        cno_df.loc[mask]
    cno = []
    non_cno = []
    card9cre = []
    for name in name_list:
        #if 'LF30_F2' in name:
        #    print(f'Getting rid of this due to error test: {name}')
        #    continue
        if name in ['card9-ca-3', 'card9-ca-4', 'card9-1', 'card9-2', 'card9-ca-4-retest']:
            date = '12-28-23'
            name = f'{date}_{name}'
        elif 'CARD9Cre' in name:
            card9cre.append(name)
            continue
        else:
            date = re.search('(.+-.+-.+?)_', name)[1]
        if date in treatment_df.index.get_level_values(2):
            cno.append(name)
        else:
            non_cno.append(name)
    non_cno_df = sort_non_cno(non_cno, behs)
    cno_df = sort_cno(cno, behs)
    if only_good_injs:
        cno_df = only_good_injections(cno_df)
    card9cre_df = sort_card9cre(card9cre, behs)
    return non_cno_df, cno_df, card9cre_df


#non_cno_df, cno_df = arrange_data(test)
