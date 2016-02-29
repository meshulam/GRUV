from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils
from data_utils.parse_files import convert_wav_files_to_nptensor
import utils

config = utils.get_config()

wav_dir = config['wav_dir']
model_input = config['model_input']

freq = config['sample_frequency']  # sample frequency in Hz

# length of clips for training. Defined in seconds
clip_len = 10

# block sizes used for training - this defines the size of our input state
block_size = freq / 4

# Used later for zero-padding song sequences
max_seq_len = int(round((freq * clip_len) / block_size))

# Convert WAVs to frequency domain with mean 0 and standard deviation of 1
if not os.path.isfile(model_input + "_x.npy"):
    print("Converting audio to tensor. Blocksize: {}, seq len: {}".format(block_size, max_seq_len))
    convert_wav_files_to_nptensor(wav_dir, block_size, max_seq_len, model_input)
else:
    print("Found tensor files at " + model_input)

# Load up the training data
print ('Loading training data from tensors')
# x_train and y_train are tensors of size (num_train_examples, num_timesteps, num_frequency_dims)
x_train = np.load(model_input + '_x.npy')
y_train = np.load(model_input + '_y.npy')
print ('Finished loading training data. Dimensions: ' + repr(x_train.shape))

#Figure out how many frequencies we have in the data
freq_space_dims = x_train.shape[2]
hidden_dims = config['hidden_dimension_size']

#Creates a lstm network
model = network_utils.create_lstm_network(
    num_frequency_dimensions=freq_space_dims,
    num_hidden_dimensions=hidden_dims)

exit()

#Load existing weights if available
#if os.path.isfile(model_output):
#    model.load_weights(model_output)

iter_count = config['iteration_count']          # Number of iterations for training
batch_size = config['batch_size']
print('Starting training')

# We set cross-validation to 0, as cross-validation will be on different datasets
# if we reload our model between runs. The moral way to handle this is to manually split
# your data into two sets and run cross-validation after you've trained the model for some number of epochs
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    nb_epoch=iter_count,
    verbose=1,
    validation_split=0.0)
print('Training complete')

outfile_name = config['model_output'] + str(iter_count)
model.save_weights(outfile_name)
print('Saved model weights to {}, Exiting!'.format(outfile_name))
