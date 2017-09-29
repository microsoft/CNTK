encoder_type = 'FF'

options = {
    'vocab_size': 2068144 if encoder_type == 'FF' else 93822, #2546935 from tf
    'emb_dim': 280, #320 in tf
    'hidden': [300, 300, 500],
    'dropout_rate': 0.2,
    'train': 'train.sample100k.ff.ctf' if encoder_type == 'FF' else 'train.seq.ctf',
    'lr_schedule': [1.0] if encoder_type == 'FF' else [1.0],
    'mom_schedule': [0.9],
    'minibatch_size': 100 if encoder_type == 'FF' else 1200, # 5000 is also ok for LSTM
    'max_epochs': 20,
    'epoch_size': 86771143, #full data
    'num_workers': 1,
}
