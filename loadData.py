"""
Code written by Jose Garrido Ramas. Loads the trajectories in the nuscenes dataset
"""

import pickle
from nuscenes_utils.nuscenes import NuScenes
import numpy as np
from helpers import *

nusc = NuScenes(version='v0.1', dataroot='/home/jose/data', verbose=True)
pickle.dump( nusc, open( "data/nusc.p", "wb" ) )

map_data_all = load_traj_data(nusc , show = False)
np.save('data/map_data_all.npy',map_data_all)

fs = 2
split_size = 7*fs #samples
step = 4*fs #samples
map_data_split = np.array(split_data(split_size, step, map_data_all))
np.save('data/map_data_split14.npy',map_data_split)

#Let's look at the number of partitions per instance
#and at the number of trajectories
#minimun length for a trajectory to be considered moving
THRESHOLD_LENGTH = 10

instance_ID = 0
ins_offset = 0
part_no = {}
linear = 0
nolinear = 0
outlier = 0
for scene_ix in range(map_data_split.shape[0]):
    traj_partitions = []
    for partition_ix in range(len(map_data_split[scene_ix])):
        part = map_data_split[scene_ix][partition_ix]
        traj_instances= []
    
        for instance_ix in range(part.shape[0]):
            instance_ID = ins_offset +instance_ix
            traj_x = map_data_split[scene_ix][partition_ix][instance_ix, :, 0]
            traj_y = map_data_split[scene_ix][partition_ix][instance_ix, :, 1]
            traj = np.array([traj_x, traj_y]).T
            
            if(~np.any(traj_x == -999) and ~np.any(traj_y == -999)):
                if(get_length(traj)>THRESHOLD_LENGTH):
                    if instance_ID not in part_no: 
                        part_no[instance_ID]= 0
                
                    part_no[instance_ID]+=1
        
    ins_offset = instance_ID+1

print('The number of non-stopped trajectories is: ', len(part_no))
print('The average number of chunks a trajectory produces is: ', np.mean(list(part_no.values())))
