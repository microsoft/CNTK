Distributed_word_embedding's input_file format instruction:
1.train_file is normal format,in which words are separated by space.
2.word_count.cpp is a word_frequency generator on the basis of train_file.
  How to use in commandline: word_count.exe [-train_file <train_file>] [-save_vocab_file <vocab for saving>] [-min_count <number>]
3.stopwords_simple.txt is sw_file which is used to filter dictionary.        