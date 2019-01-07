"""
code written by Jose Garrido Ramas
"""
import pickle
import numpy as np

f = open('../data/maps_list.pkl', "rb")
maps_list = pickle.load(f)
f.close()

def get_spatial_tensor(point, scene_ix, n, delta, test=False):
    if(test):
        scene_ix = scene_ix + 90

    map_mask = maps_list[scene_ix]
    l =0
    delta = 20
    #spatial matrix is nxn
    spatial = map_mask[(int(point[1])-delta):(int(point[1])+delta),(int(point[0])-delta):int((point[0])+delta)]

    #Let's simplify the spatial info. If, for any square, there is any 255, this square is "non drivable"
    #  _, axes = plt.subplots(1, 1, figsize=(10, 10))
    # plt.ion()
    #  do = np.array(map_mask.mask[int(ymin):int(ymax),int(xmin):int(xmax)])
    # do1 = do[(point[1]-delta):(point[1]+delta), (point[0]-delta):(point[0]+delta)]
    #mask = Image.fromarray(do1)
    #axes.imshow(mask.resize((int(mask.size[0]), int(mask.size[1])),
            #                  resample=Image.NEAREST))
    spatial_m = np.zeros((n,n))
    step = int(np.floor(2*delta/n))
    for i in range(n):
      for j in range(n):
          square =  spatial[i*step: ((i+1)*step-1), j*step:((j+1)*step-1)]    #(1 square of points equivalent to one element in the spatial matrix)
          if(np.any(square == 0)):
              spatial_m[i,j] = 1
    spatial_m = np.reshape(spatial_m, (n*n,1))
    return spatial_m




def get_ADE(predicted_traj, true_traj, observed_length):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory 
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed (which is not taken into account
    to calculate the ADE)
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = predicted_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)
  
  
def get_FDE(pred_traj, true_traj):
  L = len(pred_traj)-1
  error = np.sqrt((pred_traj[L][0] - true_traj[L][0])**2 + (pred_traj[L][0] - true_traj[L][0])**2)
  return error

    
#linear predictor
def linear_pred(vec, n_pred):
  #find acceleration
  step = (vec[len(vec)-1]-vec[0])/(len(vec)-1)
  traj_total = np.zeros((len(vec)+n_pred,2))
  traj_total[len(traj_total)-n_pred: len(traj_total)]=  np.array([vec[len(vec)-1]+step*i for i in range(1, n_pred+1)])
  traj_total[0:len(vec)] = vec
  return traj_total

def recover_traj(vel, val0):
    #Velocity x, velocity y, val0x, val0 y 
    traj = np.zeros((vel.shape[0]+1))
    traj[0] = val0
    traj[1:] = vel
    return np.cumsum(traj)
