import sys
import os
import argparse
from .wordvocab import Vocabulary

def file_exists(src):
  return (os.path.isfile(src) and os.path.exists(src))

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train-data', required=True, help="Path of the training data.")
parser.add_argument('-v', '--vocab', default="word_vocab.txt", help="Path of the word vocabulary data.")
parser.add_argument('-s', '--vocab-size', default=50000, type=int, help="Max size of the word vocabulary.")
parser.add_argument('-i', '--train-index', default="train.index.txt", help="The path of the featurized index file.")
parser.add_argument('--max-seq', default=10000, type=int, help="The maxium length of context sequence.")
parser.add_argument('--split-to-segs', default=None, help="Split long context sequence to smaller ones with length less than --max-seq.")
args = parser.parse_args()
entity_vocab = None
word_vocab = None
if args.split_to_segs is not None:
  Vocabulary.chunk_bing_corpus(args.train_data, args.split_to_segs, args.max_seq)
else:
  if not file_exists(args.vocab):
    entity_vocab, word_vocab = Vocabulary.build_bingvocab(args.train_data, args.vocab, args.vocab_size)
  else:
    entity_vocab, word_vocab = Vocabulary.load_bingvocab(args.vocab)
  if not file_exists(args.train_index):
    Vocabulary.build_bing_corpus_index(entity_vocab, word_vocab, args.train_data, args.train_index, args.max_seq)
