"""
Code written by Jose Garrido Ramas. Loads the trajectories in the nuscenes dataset
"""

import pickle
from nuscenes_utils.nuscenes import NuScenes
import numpy as np
from helpers import *

nusc = NuScenes(version='v0.1', dataroot='dataset', verbose=True)

#As loading the NuscenesDataset takes some time, dump in in the 
#data folder for later reuse in the other functions. nusc contains
#information on the annotations and the trajectories. 
pickle.dump( nusc, open( "data/nusc.p", "wb" ) )

#function LOAD_TRAJ_DATA in helpers.py loads the vehicle positions (though it can 
#easily be modified to also load other annotations). MAP_DATA_ALL is a dictionary: 
#MAP_DATA_ALL['POS'] is a list of scenes. Each scene is represented as a numpy array
#of shape instances x samples (timesteps) x 2 (coordinates x and y). Missing values 
#are represented as -999.
map_data_all = load_traj_data(nusc)
np.save('data/map_data_all.npy',map_data_all)

#Split data in chunks of 7 seconds each using the function in helpers.py
#SPLIT_DATA
fs = 2
split_size = 7*fs #samples
step = 4*fs #samples

#MAP_DATA_SPLIT saves the data in the dataset and is a list of scenes x partitions x instances x time_steps x 2.
#Last 3 dimensions are kept in a numpy array
map_data_split = np.array(split_data(split_size, step, map_data_all))
np.save('data/map_data_split14.npy',map_data_split)

#We have finished loading the data, but now we want to understand some characteristics of it.
#We are going to find the number of trajectories, and also the number of outliers.
#Let's look at the number of partitions per instance and at the number of trajectories

#minimun length for a trajectory to be considered not stopped.
THRESHOLD_LENGTH = 10
THRESHOLD_ACCELERATION = 25

instance_ID = 0
ins_offset = 0

#PART_NO is a dictionary that associates an instance ID to the number of trajectories (partitions)
#that this instance has in MAP_DATA_SPLIT
part_no = {}

#number of outliers
outliers = 0

for scene_ix in range(map_data_split.shape[0]):
    traj_partitions = []
    for partition_ix in range(len(map_data_split[scene_ix])):
        part = map_data_split[scene_ix][partition_ix]
        traj_instances= []
    
        for instance_ix in range(part.shape[0]):
            instance_ID = ins_offset + instance_ix
            traj = map_data_split[scene_ix][partition_ix][instance_ix]
            traj_x = traj[:, 0]
            traj_y = traj[:, 1]

            #If any value in the trajectory is -999 then that trajectory is not complete 
            #and is discarded
            if(~np.any(traj_x == -999) and ~np.any(traj_y == -999)):

                #If the length is greater than the threshold we consider the vehhicle to not
                #be stopped
                if(get_length(traj)>THRESHOLD_LENGTH):

                    #If the max of the acceleration is above a certain threshold, we consider
                    #this trajectory to be an outlier
                    vel2D = np.diff(traj, 1, axis = 0)
                    vel1D = np.sqrt(np.power(vel2D[:,0],2) + np.power(vel2D[:,1],2))
                    a = np.abs(np.diff(vel1D))
                    if(np.max(a) > THRESHOLD_ACCELERATION):
                        outliers = outliers + 1

                    if instance_ID not in part_no: 
                        part_no[instance_ID]= 0
                    part_no[instance_ID]+=1
        
    ins_offset = instance_ID+1

print('The number of unique trajectories is: ', len(part_no))
print('The number of trajectories (some of them belong to the same vehicle) is: ', sum(part_no.values()))
print('Of them, we consider outliers: ', outliers)
print('The average number of chunks a trajectory produces is: ', np.mean(list(part_no.values())))
