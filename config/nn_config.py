def get_neural_net_configuration():
    nn_params = {}
    nn_params['sampling_frequency'] = 22050
    #Number of hidden dimensions.
    #For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
    nn_params['hidden_dimension_size'] = 2048
    nn_params['batch_size'] = 128
    nn_params['num_iters'] = 100
    #The weights filename for saving/loading trained models
    nn_params['model_basename'] = './audioNPWeights'
    #The model filename for the training data
    nn_params['model_file'] = './datasets/audioNP'
    #The dataset directory
    nn_params['wav_dir'] = './datasets/wav/'
    return nn_params
