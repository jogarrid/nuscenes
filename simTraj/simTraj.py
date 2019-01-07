"""
Code written by José Garrido Ramas. Generates trajectories that follow the road
but have the velocity 1D of real trajectories in the nuscenes dataset.
"""

import argparse
import numpy as np
from simTrajHelpers import *
from nuscenes_utils.nuscenes import NuScenes
from helpers import *
import pickle
import os

#For reproducibility of results
np.random.seed(100)

parser = argparse.ArgumentParser(description='Using the functions in the file simTrajHelpers.py, generates in folder data file trajs_generated.npy with the simulated trajs')

parser.add_argument('-f', '--force_loading', type=str, required = False,
                    help='If T is passed, force loading of the files in the data folder, even if they have already been loaded')

args = parser.parse_args()

#if it is
force_loading = args.force_loading == 'T'

if(not os.path.isfile('../data/all_trajs_generated.npy') or force_loading):
    #Load and save simulated trajectories
    all_trajs_generated = []
    delta= 500
    test = False
    for scene_ix in range(100):
        if(scene_ix >89):
            test = True
        trajs, do = get_lines(scene_ix, delta)
        trajs_div = divide_trajs(trajs)
        new_trajs = get_new_trajs(trajs_div,do, test)
        all_trajs_generated.append(new_trajs)
        print('Finished generating traj: ' ,scene_ix) 

    np.save('../data/all_trajs_generated.npy',all_trajs_generated)

if(not os.path.isfile('../data/maps_list.pkl') or force_loading):
    #Save list of maps in data folder
    maps = load_maps()
    pickle.dump(maps, open( "../data/maps_list.pkl", "wb" ))

