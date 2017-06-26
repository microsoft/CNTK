cd `dirname $0`
cd ../../LightRNN/
python preprocess.py -datadir ../PTB/Data -outputdir ../PTB/Allocation -vocab_file vocab.txt -alloc_file word-0.location -vocabsize 10000 -seed 0
