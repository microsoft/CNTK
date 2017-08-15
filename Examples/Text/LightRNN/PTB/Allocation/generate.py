import os

dir_path = os.path.dirname(os.path.abspath(__file__))
preprocess_file = os.path.join(dir_path, '..', '..', 'LightRNN', 'preprocess.py')
assert os.path.exists(os.path.join(dir_path, '..', 'Data'))
os.system('python {} -datadir ../Data -outputdir . -vocab_file vocab.txt -alloc_file word-0.location -vocabsize 10000 -seed 0'.format(preprocess_file))
