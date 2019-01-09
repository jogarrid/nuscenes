"""
Author Jose Garrido Ramas. Load nuscenes data, the format being 100 text files (one for each scene). Each scene 
contains the trajectory data in the format (FRAME_ID, INSTANCE_ID, X, Y, CATEGORY_NAME). 
"""

from nuscenes_utils.nuscenes import NuScenes
import numpy as np
from helpers import *

nusc = NuScenes(version='v0.1', dataroot='dataset', verbose=True)

#function in helpers.py to load all the data
map_data_all = load_traj_data(nusc, only_vehicles = False)

instance_id = 0
for scene_ix in range(len(map_data_all)):
    textFile = open('traj_format/' + str(scene_ix) +'.txt', 'w')
    #First iterate over all frames (t is the FRAME_ID)
    for t in range(map_data_all[scene_ix]['pos'].shape[1]):
        #For each sample,  iterate over all its annotations
        for i in range(map_data_all[scene_ix]['pos'].shape[0]):
            #Because of how we built the map_data_all matrix, the first annotation always 
            #corresponds to the ego vehicle

            if(i!=0):
                ins_token = map_data_all[scene_ix]['instance_tokens'][i]
                instance = nusc.get('instance', ins_token)   
                ann = nusc.get('sample_annotation', instance['first_annotation_token'])
                atribute = nusc.get('attribute',ann['attribute_tokens'][0])
                name = ann['category_name']

            else:
                name = 'vehicle.ego'

            x = np.round(map_data_all[scene_ix]['pos'][i,t,0])
            y = np.round(map_data_all[scene_ix]['pos'][i,t,1])
            
            #t is the frame id
            if(x> 0 and y >0):
                sentence = str(t)+' '+str(i)+' '+ str(x)+' '+str(y)+' '+name + '\n'
                textFile.write(sentence)
                instance_id +=1
    textFile.close()
    
