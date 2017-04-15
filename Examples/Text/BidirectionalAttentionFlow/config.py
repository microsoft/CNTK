data_config = {
    'word_size' : 20,
    'word_count_threshold' : 10,
    'char_count_threshold' : 50,
    'max_context_len' : 870,
    'max_query_len' : 65,
    'pickle_file' : 'vocabs.pkl',
}

model_config = {
    'hidden_dim' : 100,
    'char_convs' : 100,
    'char_emb_dim' : 8,
    'dropout' : 0.2,
    'highway_layers' : 2,
    'two_step'       : True,
}

training_config = {
    'minibatch_size' : 2500, # in samples
    'log_freq'       : 20,   # in minibatchs
    'epoch_size'     : 1780, # in sequences
    'max_epochs'     : 100,
    'lr'             : 0.5,
    'train_data'     : 'val.ctf',
    'val_data'       : 'val.ctf',
}