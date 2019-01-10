# nuscenes
Nuscenes trajectory prediction.
For the nuscenes import to work, follow the steps in https://github.com/nutonomy/nuscenes-devkit

You need to download the dataset and place it in the dataset folder. 

As of December 2018, tensorflow is not compatible with python 3.7. Nuscenes, however, only works in python 3.7. Therefore, to be able to use both nuscenes and tensorflow, you need to switch between two virtual environments, one with python 3.6 and one with python3.7.

The file loadData.py creates, in the folder data, the files map_data_all.npy, map_data_split.npy and nusc.p . 
map_data_all contains the trajectories for all vehicles in the dataset. Map_data_split contains the same information, only the scenes have been divided into partitions. nusc.p contains information on the annotations for the trajectories.

The file loadDataFormat.py creates the folder traj_Format which contains the trajectories in the format FRAME_ID, INSTANCE_ID, X, Y, CATEGORY_NAME.

The file helpers.py contains helper functions to load the data.

In the folder simTraj,the same file helpers.py appears, along with simTrajHelpers.py (helper functions to simulating the trajectories) and simTraj.py. The result of executing simTraj.py is all_trajs_generated.npy, containing the synthetic trajectories is generated in the data folder.

EXECUTION INSTRUCTIONS:
1st, execute load_data.py and simTraj.py. All the necessary files to carry out the LSTM prediction experiments will be created in the data folder.

To carry out the LSTM prediction experiments, execute first vanilla_train.py and then vanilla_test.py (spatial_train.py and then spatial_test.py in the case of spatial LSTM). By executing 
python vanilla_train.py --dataset simulated you train on the simulated dataset. python vanilla_train.py --dataset simulatedSmall will train on the simulated but small dataset. python vanilla_train.py --dataset real will train on 
the real nuscenes teaser dataset.

