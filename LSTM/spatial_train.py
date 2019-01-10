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
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--rnn_size', type=int, default=256,
                    help='size of RNN hidden state')
# Size of each batch parameter
parser.add_argument('--batch_size', type=int, default=50,
                    help='minibatch size')
# Length of sequence to be considered parameter
parser.add_argument('--seq_length', type=int, default=12,
                    help='RNN sequence length')
# Number of epochs parameter
parser.add_argument('--num_epochs', type=int, default=300,
                    help='number of epochs')
# Frequency at which the model should be saved parameter
parser.add_argument('--save_every', type=int, default=400,
                    help='save frequency')
# Gradient value at which it should be clipped
parser.add_argument('--grad_clip', type=float, default=10.,
                    help='clip gradients at this value')
# Learning rate parameter
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate')
# Decay rate for the learning rate parameter
parser.add_argument('--decay_rate', type=float, default=0.99,
                    help='decay rate for rmsprop')
# Dimension of the embeddings parameter
parser.add_argument('--embedding_size', type=int, default=256,
                    help='Embedding dimension for the velocity coordinates')
parser.add_argument('--obs_length', type=int, default=4,
                    help='Number of observed samples')
parser.add_argument('--pred_length', type=int, default=8,
                    help='Number of predicted samples')

parser.add_argument('--embedding_sizes', type=int, default=1024,
                    help='Embedding dimension for the spatial tensor')

parser.add_argument('--n', type=int, default=6,
                    help='Number of rows and columns in the spatial matrix')

parser.add_argument('--delta', type=int, default=20,
                    help='Spatial tensor will cover +-delta pixels from a vehicle s position')

parser.add_argument('--dataset', type=str, default='simulated',
                    help='String, if equal to simulated will use the simulated trajectories, if equal to simulatedSmall will use a few of the simulated trajectories, else it will use the real trajectories')

args = parser.parse_args()
f = open('../data/maps_list.pkl', "rb")
maps_list = pickle.load(f)
f.close()

def train(args):
    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    data_loader = DataLoader(args.batch_size, args.seq_length, args.dataset, force_preprocessing=True, testing = False, n = args.n, delta = args.delta)

    # Save the arguments int the config file
    with open(os.path.join('../modelData', 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create a Vanilla LSTM model with the arguments
    model = Model(args)

    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Initialize all the variables in the graph
        sess.run(tf.global_variables_initializer())
        # Add all the variables to the list of variables to be saved
        saver = tf.train.Saver(tf.global_variables())

        # For each epoch
        print(args.num_epochs)
        for e in range(args.num_epochs):
            # Assign the learning rate (decayed acc. to the epoch number)
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            print('learning rate: ', args.learning_rate * (args.decay_rate ** e))
            # Reset the pointers in the data loader object
            data_loader.reset_batch_pointer()
            # Get the initial cell state of the LSTM
            state = sess.run(model.initial_state)

            # For each batch in this epoch
            losst = 0
            for b in range(data_loader.num_batches):
                # Tic
                start = time.time()
                # Get the source and target data of the current batch
                # x has the source data, y has the target data
                x, y, mean_std = data_loader.next_batch()
                spatial_tensors = np.array(mean_std[:])[:,3]

                input_ = np.zeros((len(x), args.seq_length, 2))
                input_s = np.zeros((len(x), args.seq_length, args.n**2))
                for i in range(len(x)):
                  for t in range(args.seq_length):
                    input_[i,t,:]= x[i][t]
                    input_s[i,t,:] = spatial_tensors[i][t][:,0]
                y = [y[i][args.obs_length:args.seq_length] for i in range(len(y))]
                # Feed the source, target data and the initial LSTM state to the model
                feed = {model.input_data: input_,model.input_s: input_s, model.target_data: y, model.initial_state: state}

                # Fetch the loss of the model on this batch, the final LSTM state from the session
                train_loss, state, _ , pred= sess.run([model.cost, model.final_state, model.train_op, model.output], feed)
                                # Toc
                losst += train_loss
                end = time.time()
                if(b==10):
                  print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        losst/b, end - start))
                  losst=0

                # Save the model if the current epoch and batch number match the frequency
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join('save', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

train(args)








