"""
Code written by Sajjad Mozaffari, modified by José Garrido Ramas to train in the nuscenes dataset
and take into account spatial information
"""

import numpy as np
import pickle
from spatialModel import *
from helpersLSTM import *
import argparse
import time

#For reproducibility of results
np.random.seed(100)

parser = argparse.ArgumentParser()

parser.add_argument('--obs_length', type=int, default=4,
                    help='Number of observed samples')
parser.add_argument('--pred_length', type=int, default=8,
                    help='Number of predicted samples')

parser.add_argument('--n', type=int, default=6,
                    help='Number of rows and of columns in the spatial matrix')
parser.add_argument('--dataset', type=str, default='simulated',
                    help='String, if not equal to simulated will use the real trajectories')
sample_args = parser.parse_args()

# Load the saved arguments to the model from the config file
with open(os.path.join('../spatialData', 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

#with trainer.model.graph.as_default():
# Initialize TensorFlow session
tf.reset_default_graph()#SAMPLE

# Initialize with the saved args
model = Model(saved_args, True)

sess = tf.InteractiveSession()
# Initialize TensorFlow saver
saver = tf.train.Saver()

# Get the checkpoint state to load the model from
ckpt = tf.train.get_checkpoint_state('save')

print('loading model: ', ckpt.model_checkpoint_path)

if(sample_args.dataset == 'simulated'):
    map_data_split = np.load('../data/all_trajs_generated.npy')  #scenes x partitions x instances x time_steps x 2
    map_data_split = np.array([[np.array(map_data_split[i])] for i in range(100)])

else: 
    map_data_split = np.load('../data/map_data_split14.npy')


# Restore the model at the checpoint
saver.restore(sess, ckpt.model_checkpoint_path)

# Initialize the dataloader object to
# Get sequences of length obs_length+pred_length
data_loader = DataLoader(1, sample_args.pred_length + sample_args.obs_length, sample_args.dataset, force_preprocessing= True, testing = True,n=sample_args.n)

# Reset the data pointers of the data loader object
data_loader.reset_batch_pointer()

total_error = 0.
total_error_l = 0.

total_error_f = 0
total_error_lf = 0
counter = 0.
j = 0

trajs = []
info = [] #mean, std, index information in array form
for b in range(data_loader.num_batches):
    # Get the source, target data for the next batch
    x, y, mean_std = data_loader.next_batch()
    mean_std = mean_std[0]
    mean_x = mean_std[0][0]
    std_x = mean_std[0][1]
    mean_y = mean_std[1][0]
    std_y = mean_std[1][1]
    traj_real = x[0]
    # The observed part of the trajectory
    obs_traj = x[0][:sample_args.obs_length]
    
    # Get the complete trajectory with both the observed and the predicted part from the model
    complete_traj = model.sample(sess, obs_traj, mean_std, num=sample_args.pred_length)
    complete_traj_ = np.zeros((complete_traj.shape[0]+1,2))
    complete_traj_[:,0]= recover_traj(complete_traj[:,0], mean_x)
    complete_traj_[:,1] = recover_traj(complete_traj[:,1], mean_y)
    j+=1
    obs_traj = np.array(obs_traj)
    index = mean_std[2]
    print(index)
    traj_real = map_data_split[index[0]+90][index[1]][index[2],0:complete_traj_.shape[0]]
    if(get_length(traj_real) >40):
        # Compute the mean error between the predicted part and the true trajectory
        error = get_ADE(complete_traj_, traj_real, sample_args.obs_length)
        errorf = get_FDE(complete_traj_, traj_real)
              
        total_error += error
        total_error_f += errorf
        
        #now the linear predictor
        complete_traj =linear_pred(obs_traj, sample_args.pred_length)
      
        j+=1
        obs_traj = np.array(obs_traj)
        complete_traj_ = np.zeros((complete_traj.shape[0]+1,2))
        
        complete_traj_[:,0]= recover_traj(complete_traj[:,0], mean_x)
        complete_traj_[:,1] = recover_traj(complete_traj[:,1], mean_y)
        errorl = get_ADE(complete_traj_, traj_real, sample_args.obs_length)
        errorlf = get_FDE(complete_traj_, traj_real)
        total_error_l += errorl
        total_error_lf += errorlf
        print("Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories")
        info.append(mean_std)
        trajs.append(complete_traj)
  
# Print the mean error across all the batches
print("Total mean error of the model is ", total_error/data_loader.num_batches)
print("Total mean error of the linear model is ", total_error_l/data_loader.num_batches)

print("Total final displacement error of the model is ", total_error_f/data_loader.num_batches)
print("Total final displacement error of the linear model is ", total_error_lf/data_loader.num_batches)
