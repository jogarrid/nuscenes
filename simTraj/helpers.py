"""
Code written by Jose Garrido Ramas. HELPER functions to load and work with 
the nuscenes trajectories data.
"""


import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.axes import Axes
#get_ipython().run_line_magic('matplotlib', 'inline')
from nuscenes_utils.nuscenes import NuScenes
import numpy as np


def get_length(traj):
    """
    Get length (pixels) of trajectory traj. The complication arises in that 
    #the length of the trajectory is not proportional to the number of points (point density varies)
    """

    #First sample trajectory to avoid noise in getting the length. 
    step = max(int(len(traj)/10),1)
    t = np.arange(0, len(traj), step)
    if(len(traj)>1):
        delta_x = np.diff(traj[t,0])
        delta_y = np.diff(traj[t,1])   
        return (sum(np.sqrt(np.power(delta_x, 2)+np.power(delta_y, 2))))
    else: 
        return 0


def load_traj_data(nusc):
    records = nusc.scene

    #MAP_DATA_ALL['POS'] is a list of scenes. Each scene is represented as a numpy array
    #of shape instances x samples (timesteps) x 2 (coordinates x and y). Missing values 
    #are represented as -999.
    map_positions_all = []
    
    for i, record in enumerate(records):
        scene_token = record['token']
        no_samples = record['nbr_samples']

        log_record = nusc.get('log', record['log_token'])
        map_record = nusc.get('map', log_record['map_token'])

        # map_record['mask'].mask holds a MapMask instance that we need below.
        map_mask = map_record['mask']

        #First dimension indicates instance; second dimension indicates sample no (or time step); third dimension is x and y map
        #position in pixels
        map_positions = {'pos':[], 'instance_tokens':[]}
        map_positions['pos'] = np.ones((1,no_samples,2))*(-999)  #-999 indicates missing value
	
        sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)

        map_positions['instance_tokens'].append('Doesnt apply; ego vehicle')

        for i in range(0,len(sample_tokens)):
            sample_record = nusc.get('sample', sample_tokens[i])

            # Poses are associated with the sample_data. Here we use the LIDAR_TOP sample_data.
            sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])

            pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

            # Recover the ego pose. A 1 is added at the end to make it homogenous coordinates.
            pose = np.array(pose_record['translation'] + [1])

            # Calculate the pose on the map.
            map_pose = np.dot(map_mask.transform_matrix, pose)

            # Ego vehicle trajectory is saved on first row of map_positions
            map_positions['pos'][0][i][0] = map_pose[0]
            map_positions['pos'][0][i][1] = map_pose[1]

            #Obtain all annotations for a sample
            annotations = sample_record['anns']


            for ann_token in annotations:
                ann = nusc.get('sample_annotation', ann_token) 
                instance = nusc.get('instance', ann['instance_token'])   

                #If this is the first annotation for this instance, and it is not a parked vehicle, we save its trajectory 
                #(else, it already has been done, or we are not interested in doing so)
                if(instance['first_annotation_token'] == ann['token']  and len(ann['attribute_tokens']) > 0):
                    atribute = nusc.get('attribute',ann['attribute_tokens'][0])

                    #We only save the trajectory of vehicles that are not annotated as parked
                    if(atribute['name'] != 'vehicle.parked' and atribute['name'].split('.')[0] == 'vehicle'):
                        #First, all the values are -999. We then fill in the values of this instance's trajectory.
                        #The not filled in values (missing values) will then be -999.
                        map_positions['pos'] = np.concatenate((map_positions['pos'],-999*np.ones((1,no_samples,2))))
                        map_positions['instance_tokens'].append(ann['instance_token'])
                        instance_index = map_positions['pos'].shape[0]-1
                        j = i
                        ann_record = ann
                        pose = np.array(ann_record['translation']+ [1])
                        #Transform pose from 3D coordinates to 2D pixels in the map
                        map_pose = np.dot(map_mask.transform_matrix, pose)
                        map_positions['pos'][instance_index][j][0] = map_pose[0]
                        map_positions['pos'][instance_index][j][1] = map_pose[1]

                        #While this instance is annotated in the following time step, save its position
                        while not ann_record['next'] == "":
                            ann_record = nusc.get('sample_annotation', ann_record['next'])
                            pose = np.array(ann_record['translation'] + [1])
                            map_pose = np.dot(map_mask.transform_matrix, pose)
                            j += 1
                            map_positions['pos'][instance_index][j][0] = map_pose[0]
                            map_positions['pos'][instance_index][j][1] = map_pose[1]

        map_positions_all.append(map_positions)
        
    return map_positions_all

def split_data(split_size, step, map_data_all):
    """
    split MAP_DATA_ALL in partitions of duration SPLIT_SIZE (samples) and step between them STEP
    """
    map_data_split = [] #First dimension indicates scene; second dimension,partition. 
    #other dimensions are a matrix which keeps the value of the positions and is of size instances x time x 2.
    #In list instance_tokens, first dimension indicates scene, second indicates token for a certain instance

    fs = 2
    for map_1scene_data in map_data_all:
        partitions = []  #partitions for a specific scene
        i = 0
        while(i+split_size <= map_1scene_data['pos'].shape[1]):
            partition = map_1scene_data['pos'][:,i:(i+split_size)]
            i = i + step
            partitions.append(partition)

        map_data_split.append(partitions)

    return map_data_split


def plot_2traj_partition(nusc, record, partition, part_pred, instance_tokens):  
    """
    Show in map all the predicted trajectories along with the real ones. 
    """
    scene_token = record['token']
    no_samples = record['nbr_samples']

    log_record = nusc.get('log', record['log_token'])
    map_record = nusc.get('map', log_record['map_token'])

    # map_record['mask'].mask holds a MapMask instance that we need below.
    map_mask = map_record['mask']

    # First, draw the map mask
    _, axes = plt.subplots(1, 1, figsize=(10, 10))
    plt.ion()

    #Find xmin xmax ymin and ymax in order to know how much of the map we
    #need to show.
    xmin = 1000000
    xmax = 0
    ymin = 1000000
    ymax = 0 
    for i in range(partition.shape[1]):
        indexes = np.where(partition[:,i,0] != -999)[0]
        for instance_index in indexes:
            x = partition[instance_index][i][0]
            y = partition[instance_index][i][1]
            xmin = min(xmin,x) 
            xmax = max(xmax,x)
            ymin = min(ymin,y)
            ymax = max(ymax,y)

    delta = 100
    do = np.array(map_mask.mask[int(ymin):int(ymax),int(xmin):int(xmax)])
    mask = Image.fromarray(do)
    axes.imshow(mask.resize((int(mask.size[0]), int(mask.size[1])),
                            resample=Image.NEAREST))

    #blue for observed. Green for real. Red for predicted. 
    #Let's figure out which instances have all data available (no x or y is -999)
    indexes = []
    for p in range(part_pred.shape[0]):
        if(not np.any(part_pred[p,0:9]== -999)):
            indexes.append(p)
    j = 0
    i = 0
    n_obs = 4
    #plot each instance the index of which is in INDEXES
    for instance_index in indexes:
        x = partition[instance_index,0:9,0] - xmin 
        y = partition[instance_index,0:9,1] - ymin 
        xpred = part_pred[instance_index,:,0] - xmin 
        ypred = part_pred[instance_index,:,1] - ymin 

        #We represent the ego vehicle as a star
        if(instance_index == 0): 
            axes.plot(x[0:(n_obs+1)],y[0:(n_obs+1)],color = 'b',marker="*", markersize = 7)
            axes.plot(xpred[n_obs:],ypred[n_obs:],color = 'r',marker="*", markersize = 7)
            axes.plot(x[n_obs:],y[n_obs:],color = 'g',marker="*", markersize = 7)

        else:
            instance_token = instance_tokens[instance_index]
            instance = nusc.get('instance', instance_token)
            cat = nusc.get('category', instance['category_token'])
            #We represent humans as bars
            if(cat['name'].split('.')[0] == 'human'):
                axes.plot(x[0:(n_obs+1)],y[0:(n_obs+1)],color = 'b',marker="|", markersize = 7)
                axes.plot(xpred[n_obs:],ypred[n_obs:],color = 'r',marker="|", markersize = 7)
                axes.plot(x[n_obs:],y[n_obs:],color = 'g',marker="|", markersize = 7)
            
            #We represent cars as squares
            elif(cat['name'].split('.')[0] == 'vehicle'):
                axes.plot(x[0:(n_obs+1)],y[0:(n_obs+1)],color = 'b',marker="s", markersize = 5)
                axes.plot(xpred[n_obs:],ypred[n_obs:],color = 'r',marker="s", markersize = 5)
                axes.plot(x[n_obs:],y[n_obs:],color = 'g',marker="s", markersize = 5)
                

def plot_1traj_partition(nusc, record, partition, instance_tokens): 
    """
    Show in map all the trajectories in a scene 
    """ 
    scene_token = record['token']
    no_samples = record['nbr_samples']

    log_record = nusc.get('log', record['log_token'])
    map_record = nusc.get('map', log_record['map_token'])

    # map_record['mask'].mask holds a MapMask instance that we need below.
    map_mask = map_record['mask']

    # First, draw the map mask
    _, axes = plt.subplots(1, 1, figsize=(10, 10))
    plt.ion()

    xmin = 1000000
    xmax = 0
    ymin = 1000000
    ymax = 0 
    for i in range(partition.shape[1]):
        indexes = np.where(partition[:,i,0] != -999)[0]
        for instance_index in indexes:
            x = partition[instance_index][i][0]
            y = partition[instance_index][i][1]
            xmin = min(xmin,x) 
            xmax = max(xmax,x)
            ymin = min(ymin,y)
            ymax = max(ymax,y)


    delta = 100
    do = np.array(map_mask.mask[(int(ymin)-delta):(int(ymax)+delta),(int(xmin)-delta):(int(xmax)+delta)])
    mask = Image.fromarray(do)
    axes.imshow(mask.resize((int(mask.size[0]), int(mask.size[1])),
                            resample=Image.NEAREST))

    #Let's figure out which instances have all data available (no x or y is -999)
    indexes = []
    for p in range(partition.shape[0]):
        if(not np.any(partition[p,:]== -999)):
            indexes.append(p)
    j = 0
    i = 0
    n_obs = 4
    for instance_index in indexes:
        x = partition[instance_index,:,0] - xmin +delta
        y = partition[instance_index,:,1] - ymin +delta
        if(instance_index == 0): #ego vehicle
            axes.plot(x,y,color = 'b',marker="*", markersize = 7)

        else:
            instance_token = instance_tokens[instance_index]
            instance = nusc.get('instance', instance_token)
            cat = nusc.get('category', instance['category_token'])
            if(cat['name'].split('.')[0] == 'human'):
                axes.plot(x,y,color = 'b',marker="|", markersize = 7)
            elif(cat['name'].split('.')[0] == 'vehicle'):
                #we only plot the vehicle if it is stopped of moving, not if it is parked
                axes.plot(x,y,color = 'b',marker="s", markersize = 5)


