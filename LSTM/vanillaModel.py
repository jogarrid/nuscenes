import os
import pickle
import numpy as np
import random
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell
from helpersLSTM import *


class DataLoader():
    def __init__(self, batch_size=50, seq_length=5, dataset= 'simulated', force_preprocessing = True, testing=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : RNN sequence length
        '''

        # Data directory where the pre-processed pickle file resides
        self.data_dir = '../modelData/'

        # Store the batch size and the sequence length arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path of the file in which the data needs to be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        self.testing = testing
        
        # If the file doesn't exist already or if forcePreProcess is true
        if not(os.path.exists(data_file)) or force_preprocessing:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files
            self.preprocess(dataset, data_file)

        # Load the data from the pickled file
        self.load_preprocessed(data_file)
        # Reset all the pointers
        self.reset_batch_pointer()

    def preprocess(self, dataset, data_file):
        '''
        scenes : List of scene indixes to be loaded
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_ins_data would be a dictionary with mapping from each instance to their
        # trajectories given by matrix 3 x numPoints with each column
        # in the order x, y, frameId

        all_ins_data = {}
        dataset_indices = []
        current_ins = 0
        threshold = 2
        # where each column is a (frameId, instance_Id, y, x) vector
        # where each column is a (frameId, instance_Id, y, x) vector
        if(dataset == 'simulated'):
            print('USING THE SIMULATED TRAJECTORIES')
            data_all = np.load('../data/all_trajs_generated.npy')  #scenes x partitions x instances x time_steps x 2
            data_all = np.array([[np.array(data_all[i])] for i in range(100)])
        elif(dataset == 'simulatedSmall'):
            print('USING THE SIMULATED TRAJECTORIES, VERSION WITH FEW TRAJECTORIES')
            data_all = np.load('../data/few_trajs_generated.npy')  #scenes x partitions x instances x time_steps x 2
            data_all = np.array([[np.array(data_all[i])] for i in range(100)])
        else: 
            data_all = np.load('../data/map_data_split14.npy')
        
        if(self.testing):
            scenes = list(range(90,100))
        else: 
            scenes = list(range(90))
        #If we are training, we make the dataset more balanced
        data_all = data_all[scenes]
        threshold_stopped = 1
        threshold_linear = 2

      
        frame_Id = 0
        instance_Id = 0
        data = []

        means = []
        stds = []
        indexes = []
        self.mean_std = []
        trajs_nl = 0
        scenes_nl = []

        for scene_ix in range(data_all.shape[0]):
          for partition_ix in range(len(data_all[scene_ix])):
            part = data_all[scene_ix][partition_ix]
            for instance_ix in range(part.shape[0]):
              traj_x = data_all[scene_ix][partition_ix][instance_ix, :, 0]
              traj_y = data_all[scene_ix][partition_ix][instance_ix, :, 1]

              if(~np.any(traj_x == -999) and ~np.any(traj_y == -999)):
                #create new instance trajectories by rotating the ones we have
                if(not self.testing):
                    r = np.random.random() #to make the dataset more balanced, we take out all non moving vehicles and 30% of linear trajectories
                    trajs_nl += 1
                    scenes_nl.append(scene_ix)

                    angles = list(np.arange(0,2*np.pi,2*np.pi/3))
                    traj_x_or = np.copy(traj_x)
                    traj_y_or = np.copy(traj_y)
                    for angle in angles:
                      traj_x = traj_x_or*np.cos(angle)-traj_y_or*np.sin(angle)
                      traj_y = traj_y_or*np.cos(angle)+traj_x_or*np.sin(angle)

                      mean_x = traj_x[0]
                      mean_y = traj_y[0]
                      std_x = np.std(traj_x[0:4])
                      std_y = np.std(traj_y[0:4])
                      #traj_x = (traj_x - mean_x)/(std_x + 0.5)
                      #traj_y = (traj_y - mean_y)/(std_y +0.5)
                      traj_x = np.diff(traj_x)
                      traj_y = np.diff(traj_y)
                      means.append([mean_x, mean_y])
                      stds.append([std_x, std_y])
                      indexes.append([scene_ix, partition_ix, instance_ix])
                      for time_ix in range(traj_x.shape[0]):
                        data.append([frame_Id, instance_Id, traj_x[time_ix], traj_y[time_ix]])
                      self.mean_std.append([[mean_x, std_x], [mean_y, std_y], [scene_ix, partition_ix, instance_ix]])
                      instance_Id += 1 
                else:
                  mean_x = np.mean(traj_x[0])
                  mean_y = np.mean(traj_y[0])
                  std_x = np.std(traj_x[0:4])
                  std_y = np.std(traj_y[0:4])
                  #if(std_x > threshold or std_y>threshold):
                  traj_x = np.diff(traj_x)
                  traj_y = np.diff(traj_y)
                  means.append([mean_x, mean_y])
                  stds.append([std_x, std_y])
                  indexes.append([scene_ix, partition_ix, instance_ix])

                  for time_ix in range(traj_x.shape[0]):
                    data.append([frame_Id, instance_Id, traj_x[time_ix], traj_y[time_ix]])
                  self.mean_std.append([[mean_x, std_x], [mean_y, std_y], [scene_ix, partition_ix, instance_ix]])
                  instance_Id += 1    
            frame_Id += 1
        data = np.array(data).T
        # Get the number of pedestrians in the current dataset
        numIns = np.size(np.unique(data[1, :]))  #an instance can be a car, a pedestrian, a bycicle, etc...

        for instance_Id in range(numIns):
            # Extract trajectory of the current instance
            traj = data[:, data[1, :] == instance_Id]
                        
            # Format it as (x, y, frameId)
            traj = traj[[2, 3, 0], :]

            # Store this in the dictionary
            all_ins_data[instance_Id] = traj

            mean_std = self.mean_std[instance_Id]
            mean_x = mean_std[0][0]
            std_x = mean_std[0][1]
            mean_y = mean_std[1][0]
            std_y = mean_std[1][1]
            
           # traj_r = np.zeros(traj.shape)
            #traj_r[0,:] =  traj[0,:] +mean_x
           #traj_r[1,:] = traj[1,:] +mean_y
            index = indexes[instance_Id]

        # Current dataset done
        dataset_indices.append(current_ins+numIns)

        # The complete data is a tuple of all intance data, and dataset ped indices
        complete_data = (all_ins_data, dataset_indices)
        # Store the complete data into the pickle file
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : The path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()

        # Get the data from the pickle file
        all_ins_data = self.raw_data[0]

        # Construct the data with sequences(or trajectories) longer than seq_length
        self.data = []
        counter = 0

        # For each instance in the data
        
        for instance_Id in all_ins_data:
            # Extract his trajectory
            traj = all_ins_data[instance_Id]
            
            # If the length of the trajectory is greater than seq_length 
            if traj.shape[1] > (self.seq_length):
                self.data.append(traj[[0, 1], :].T)
                counter += 1

        #shuffle the data (if training) so all trajectories belonging to an instance won´t be in the same batch in different epoch
        indexes = np.arange(len(self.data))
        self.data = np.array(self.data)
        self.mean_std = np.array(self.mean_std)
        
        #if training, shuffle the data
        if(not self.testing): 
          np.random.shuffle(indexes)
        
        self.data = self.data[indexes]
        self.mean_std = self.mean_std[indexes]
                
        # Calculate the number of batches (each of batch_size) in the data
        self.num_batches = int(counter / self.batch_size)

        
    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        # List of source and target data for the current batch
        x_batch = []
        y_batch = []
        # For each sequence in the batch
        for i in range(self.batch_size):
            # Extract the trajectory of the pedestrian pointed out by self.pointer
            traj = self.data[self.pointer]
            mean_std = self.mean_std[self.pointer]
            idx = 0 
            # Append the trajectory from idx until seq_length into source and target data
            x_batch.append(np.copy(traj[idx:idx+self.seq_length, :]))
            y_batch.append(np.copy(traj[idx+1:idx+self.seq_length+1, :]))
            self.tick_batch_pointer()

        return x_batch, y_batch, mean_std

    def tick_batch_pointer(self):
        '''
        Advance the data pointer
        '''
      
        #TODO: randomize data as you set the pointer to 0 so that you don´t train always over the same batches
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0
            indexes = np.arange(len(self.data))
            self.data = np.array(self.data)
            self.mean_std = np.array(self.mean_std)

            #if training, shuffle the data
            if(not self.testing): 
              np.random.shuffle(indexes)

            self.data = self.data[indexes]
            self.mean_std = self.mean_std[indexes]

    def reset_batch_pointer(self):
        '''
        Reset the data pointer
        '''
        self.pointer = 0

# The Vanilla LSTM model
class Model():
    def __init__(self, args, infer=False):
        '''
        I
        nitialisation function for the class Model.
        Params:
        args: Contains arguments required for the Model creation
        '''

        # If sampling new trajectories, then infer mode
        if infer:
            # Infer one position at a time
            args.batch_size = 1
            args.seq_length = 1

        # Store the arguments
        self.args = args

        # Initialize a BasicLSTMCell recurrent unit
        # args.rnn_size contains the dimension of the hidden state of the LSTM
        cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units = args.rnn_size)

        # TODO: (improve) Dropout layer can be added here
        # Store the recurrent unit
        self.cell = cell

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # Input data contains sequence of (x,y) points
        self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, 2])
        # target data contains sequences of (x,y) points as well
        self.target_data = tf.placeholder(tf.float32, [None, args.pred_length, 2])

        # Learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # Initial cell state of the LSTM (initialised with zeros)
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # Output size is the set of parameters (mu, sigma, corr)
        output_size = 5  # 2 mu, 2 sigma and 1 corr

        # Embedding for the spatial coordinates
        with tf.variable_scope("coordinate_embedding"):
            #  The spatial embedding using a ReLU layer
            #  Embed the 2D coordinates into embedding_size dimensions
            #  TODO: (improve) For now assume embedding_size = rnn_size
            embedding_w = tf.get_variable("embedding_w", [2, args.embedding_size])
            embedding_b = tf.get_variable("embedding_b", [args.embedding_size])

        # Output linear layer
        with tf.variable_scope("rnnlm"):
            output_w = tf.get_variable("output_w", [args.rnn_size, output_size], initializer=tf.truncated_normal_initializer(stddev=0.01), trainable=True)
            output_b = tf.get_variable("output_b", [output_size], initializer=tf.constant_initializer(0.01), trainable=True)

        # Split inputs according to sequences.
        inputs = tf.split(self.input_data, args.seq_length, 1)
        # Get a list of 2D tensors. Each of size numPoints x 2
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # Embed the input spatial points into the embedding space
        embedded_inputs = []
        for x in inputs:
            # Each x is a 2D tensor of size numPoints x 2
            # Embedding layer
            embedded_x = tf.nn.relu(tf.add(tf.matmul(x, embedding_w), embedding_b))
            embedded_inputs.append(embedded_x)

        # Feed the embedded input data, the initial state of the LSTM cell, the recurrent unit to the seq2seq decoder
        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(embedded_inputs, self.initial_state, self.cell, loop_function=None)

        #To ensure that we only do backprogration in the predicted samples, at training 
        #discard the outputs corresponding to the observed samples
        if(len(outputs)>1):
          outputs = outputs[args.obs_length:args.seq_length]
        
        # Concatenate the outputs from the RNN decoder and reshape it to ?xargs.rnn_size
        output = tf.reshape(tf.concat( outputs,1), [-1, args.rnn_size])

        # Apply the output linear layer
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        
        # Store the final LSTM cell state after the input data has been feeded
        self.final_state = last_state

        # reshape target data so that it aligns with predictions
        flat_target_data = tf.reshape(self.target_data, [-1, 2])
        # Extract the x-coordinates and y-coordinates from the target data
        [x_data, y_data] = tf.split(flat_target_data, 2, 1)

        def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
            '''
            Function that implements the PDF of a 2D normal distribution
            params:
            x : input x points
            y : input y points
            mux : mean of the distribution in x
            muy : mean of the distribution in y
            sx : std dev of the distribution in x
            sy : std dev of the distribution in y
            rho : Correlation factor of the distribution
            '''
            # eq 3 in the paper
            # and eq 24 & 25 in Graves (2013)
            # Calculate (x - mux) and (y-muy)
            
            normx = tf.subtract(x, mux)
            normy = tf.subtract(y, muy)
            # Calculate sx*sy
            sxsy = tf.multiply(sx, sy)
            # Calculate the exponential factor
            z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
            negRho = 1 - tf.square(rho)
            # Numerator
            result = tf.exp(tf.div(-z, 2*negRho))
            # Normalization constant
            denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
            # Final PDF calculation
            result = tf.div(result, denom)
            self.result = result
            return result

        # Important difference between loss func of Social LSTM and Graves (2013)
        # is that it is evaluated over all time steps in the latter whereas it is
        # done from t_obs+1 to t_pred in the former
        
        def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
            '''
            Function to calculate given a 2D distribution over x and y, and target data
            of observed x and y points
            params:
            z_mux : mean of the distribution in x
            z_muy : mean of the distribution in y
            z_sx : std dev of the distribution in x
            z_sy : std dev of the distribution in y
            z_rho : Correlation factor of the distribution
            x_data : target x points
            y_data : target y points
            '''
            step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

            # Calculate the PDF of the data w.r.t to the distribution
            result0_1 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_2 = tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_3 = tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_4 = tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)

            result0 = tf.div(tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4), tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
            result0 = tf.multiply(tf.multiply(result0, step), step)

            # For numerical stability purposes
            epsilon = 1e-20

            # TODO: (resolve) I don't think we need this as we don't have the inner
            # summation
            # result1 = tf.reduce_sum(result0, 1, keep_dims=True)
            # Apply the log operation
            result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

            # TODO: For now, implementing loss func over all time-steps
            # Sum up all log probabilities for each data point
            return tf.reduce_sum(result1)

        def get_coef(output):
            # eq 20 -> 22 of Graves (2013)
            # TODO : (resolve) Does Social LSTM paper do this as well?
            # the paper says otherwise but this is essential as we cannot
            # have negative standard deviation and correlation needs to be between
            # -1 and 1

            z = output
            # Split the output into 5 parts corresponding to means, std devs and corr
            z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, 1)

            # The output must be exponentiated for the std devs
            z_sx = tf.exp(z_sx)
            z_sy = tf.exp(z_sy)
            # Tanh applied to keep it in the range [-1, 1]
            z_corr = tf.tanh(z_corr)

            return [z_mux, z_muy, z_sx, z_sy, z_corr]
        
        # Extract the coef from the output of the linear layer
        [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(output)
        # Store the output from the model
        self.output = output

        # Store the predicted outputs
        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr

        # Compute the loss function
        lossfunc = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)

        # Compute the cost
        self.cost = tf.div(lossfunc, (args.batch_size * args.seq_length))

        # Get trainable_variables
        tvars = tf.trainable_variables()

        # TODO: (resolve) We are clipping the gradients as is usually done in LSTM
        # implementations. Social LSTM paper doesn't mention about this at all
        # Calculate gradients of the cost w.r.t all the trainable variables
        self.gradients = tf.gradients(self.cost, tvars)
        # Clip the gradients if they are larger than the value given in args
        grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        # NOTE: Using RMSprop as suggested by Social LSTM instead of Adam as Graves(2013) does
        # optimizer = tf.train.AdamOptimizer(self.lr)
        # initialize the optimizer with teh given learning rate
        optimizer = tf.train.RMSPropOptimizer(self.lr)

        # Train operator
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, traj, num=10):
        '''
        Given an initial trajectory (as a list of tuples of points), predict the future trajectory
        until a few timesteps
        Params:
        sess: Current session of Tensorflow
        traj: List of past trajectory points
        num: Number of time-steps into the future to be predicted
        '''
        def sample_gaussian_2d(mux, muy, sx, sy, rho):
            '''
            Function to sample a point from a given 2D normal distribution
            params:
            mux : mean of the distribution in x
            muy : mean of the distribution in y
            sx : std dev of the distribution in x
            sy : std dev of the distribution in y
            rho : Correlation factor of the distribution
            '''
            # Extract mean
            mean = [mux, muy]
            # Extract covariance matrix
            cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
            # Sample a point from the multivariate normal distribution
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        # Initial state with zeros
        state = sess.run(self.cell.zero_state(1, tf.float32))

        # Iterate over all the positions seen in the trajectory
        for pos in traj[:-1]:
            # Create the input data tensor
            data = np.zeros((1, 1, 2), dtype=np.float32)
            data[0, 0, 0] = pos[0]  # x
            data[0, 0, 1] = pos[1]  # y

            # Create the feed dict
            feed = {self.input_data: data, self.initial_state: state}
            # Get the final state after processing the current position
            [state] = sess.run([self.final_state], feed)

        ret = traj

        # Last position in the observed trajectory
        last_pos = traj[-1]

        # Construct the input data tensor for the last point
        prev_data = np.zeros((1, 1, 2), dtype=np.float32)
        prev_data[0, 0, 0] = last_pos[0]  # x
        prev_data[0, 0, 1] = last_pos[1]  # y

        for t in range(num):
            # Create the feed dict
            feed = {self.input_data: prev_data, self.initial_state: state}

            # Get the final state and also the coef of the distribution of the next point
            [o_mux, o_muy, o_sx, o_sy, o_corr, state] = sess.run([self.mux, self.muy, self.sx, self.sy, self.corr, self.final_state], feed)

            # Sample the next point from the distribution
            next_x, next_y = sample_gaussian_2d(o_mux[0][0], o_muy[0][0], o_sx[0][0], o_sy[0][0], o_corr[0][0])
            # Append the new point to the trajectory
            ret = np.vstack((ret, [next_x, next_y]))

            # Set the current sampled position as the last observed position
            prev_data[0, 0, 0] = next_x
            prev_data[0, 0, 1] = next_y

        return ret
      
      




