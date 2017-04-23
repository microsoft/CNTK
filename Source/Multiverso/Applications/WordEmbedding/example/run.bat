set size=300
set text=(train_file's name,e.g. enwiki2014)
set read_vocab=(vocab's Directory string,e.g. "C:\Users\Leif\dataset\enwiki2014_vocab.txt")
set train_file=(train_file's Directory string,e.g. "C:\Users\Leif\dataset\enwiki2014")
set binary=1
set cbow=1
set alpha=0.01
set epoch=20
set window=5
set sample=0
set hs=0
set negative=5
set threads=16
set mincount=5
set sw_file=stopwords_simple.txt
set stopwords=0
set data_block_size=1000000000
set max_preload_data_size=20000000000
set use_adagrad=0
set is_pipeline=0
set output=%text%_%size%.bin

distributed_word_embedding.exe -max_preload_data_size %max_preload_data_size% -is_pipeline %is_pipeline% -alpha %alpha% -data_block_size %data_block_size% -train_file %train_file% -output %output% -threads %threads% -size %size% -binary %binary% -cbow %cbow% -epoch %epoch% -negative %negative% -hs %hs% -sample %sample% -min_count %mincount% -window %window% -stopwords %stopwords% -sw_file %sw_file% -read_vocab %read_vocab% -use_adagrad %use_adagrad% 

