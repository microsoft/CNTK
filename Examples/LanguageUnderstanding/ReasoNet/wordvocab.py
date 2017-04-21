import sys
import os
import math
import functools
import numpy as np

class WordFreq:
  def __init__(self, word, id, freq):
    self.word = word
    self.id = id
    self.freq = freq

class Vocabulary:
  """Build word vocabulary with frequency"""
  def __init__(self, name):
    self.name = name
    self.size = 0
    self.__dict = {}
    self.__has_index = False
    self.__reverse = {}

  def push(self, word):
    if word in self.__dict:
      self.__dict[word].freq += 1
    else:
      self.__dict[word] = WordFreq(word, len(self.__dict), 1)

  def build_index(self, max_size):
    def word_cmp(x, y):
      if x.freq == y.freq :
        return (x.word > y.word) - (x.word < y.word)
      else:
        return x.freq - y.freq

    items = sorted(self.__dict.values(), key=functools.cmp_to_key(word_cmp), reverse=True)
    if len(items)>max_size:
      del items[max_size:]
    self.size=len(items)
    self.__dict.clear()
    for it in items:
      it.id = len(self.__dict)
      self.__dict[it.word] = it
    self.__has_index = True

  def save(self, dst):
    if not self.__has_index:
      self.build_index(sys.maxsize)
    if self.name != None:
      dst.write("{0}\t{1}\n".format(self.name, self.size))
    for it in sorted(self.__dict.values(), key=lambda it:it.id):
      dst.write("{0}\t{1}\t{2}\n".format(it.word, it.id, it.freq))

  def load(self, src):
    line = src.readline()
    if line == "":
      return
    line = line.rstrip('\n')
    head = line.split()
    max_size = sys.maxsize
    if len(head) == 2:
      self.name = head[0]
      max_size = int(head[1])
    cnt = 0
    while cnt < max_size:
      line = src.readline()
      if line == "":
        break
      line = line.rstrip('\n')
      items = line.split()
      self.__dict[items[0]] = WordFreq(items[0], int(items[1]), int(items[2]))
      cnt += 1
    self.size = len(self.__dict)
    self.__has_index = True

  def __getitem__(self, key):
    if key in self.__dict:
      return self.__dict[key]
    else:
      return None

  def values(self):
    return self.__dict.values()

  def __len__(self):
    return self.size

  def __contains__(self, q):
    return q in self.__dict

  def lookup_by_id(self, id):
    if self.__reverse is None or len(self.__reverse) == 0:
      self.__reverse = {}
      for item in self.__dict.items():
        self.__reverse[item[1].id]=item[0]
    return self.__reverse[id]

  @staticmethod
  def is_cnn_entity(word):
    return word.startswith('@entity') or word.startswith('@placeholder')

  @staticmethod
  def load_vocab(vocab_src):
    """
    Loa vocabulary from file.

    Args:
      vocab_src (`str`): the file stored with the vocabulary data
      
    Returns:
      :class:`Vocabulary`: Vocabulary of the entities
      :class:`Vocabulary`: Vocabulary of the words
    """
    word_vocab = Vocabulary("WordVocab")
    entity_vocab = Vocabulary("EntityVocab")
    with open(vocab_src, 'r', encoding='utf-8') as src:
      entity_vocab.load(src)
      word_vocab.load(src)
    return entity_vocab, word_vocab

  @staticmethod
  def build_vocab(input_src, vocab_dst, max_size=50000):
    """
    Build vocabulary from raw corpus file.

    Args:
      input_src (`str`): the path of the corpus file
      vocab_dst (`str`): the path of the vocabulary file to save the built vocabulary
      max_size (`int`): the maxium size of the word vocabulary
    Returns:
      :class:`Vocabulary`: Vocabulary of the entities
      :class:`Vocabulary`: Vocabulary of the words
    """
    # Leave the first as Unknown
    max_size -= 1
    word_vocab = Vocabulary("WordVocab")
    entity_vocab = Vocabulary("EntityVocab")
    linenum = 0
    print("Start build vocabulary from {0} with maxium words {1}. Saved to {2}".format(input_src, max_size, vocab_dst))
    with open(input_src, 'r', encoding='utf-8') as src:
      all_lines = src.readlines()
      print("Total lines to process: {0}".format(len(all_lines)))
      for line in all_lines:
        line = line.strip('\n')
        ans, query_words, context_words = Vocabulary.parse_corpus_line(line)
        for q in query_words:
          if Vocabulary.is_cnn_entity(q):
          #if q.startswith('@'):
            entity_vocab.push(q)
          else:
            word_vocab.push(q)
        for q in context_words:
          #if q.startswith('@'):
          if Vocabulary.is_cnn_entity(q):
            entity_vocab.push(q)
          else:
            word_vocab.push(q)
        linenum += 1
        if linenum%1000==0:
          sys.stdout.write(".")
          sys.stdout.flush()
    print()
    entity_vocab.build_index(max_size)
    word_vocab.build_index(max_size)
    with open(vocab_dst, 'w', encoding='utf-8') as dst:
      entity_vocab.save(dst)
      word_vocab.save(dst)
    print("Finished to generate vocabulary from: {0}".format(input_src))
    return entity_vocab, word_vocab

  @staticmethod
  def parse_corpus_line(line):
    """
    Parse bing corpus line to answer, query and context.

    Args:
      line (`str`): A line of text of bing corpus
    Returns:
      :`str`: Answer word
      :`str[]`: Array of query words
      :`str[]`: Array of context/passage words

    """
    data = line.split('\t')
    query = data[0]
    answer = data[1]
    context = data[2]
    query_words = query.split()
    context_words = context.split()
    return answer, query_words, context_words

  def build_corpus(entities, words, corpus, output, max_seq_len=100000):
    """
    Build featurized corpus and store it in CNTK Text Format.

    Args:
      entities (class:`Vocabulary`): The entities vocabulary
      words (class:`Vocabulary`): The words vocabulary
      corpus (`str`): The file path of the raw corpus
      output (`str`): The file path to store the featurized corpus data file
    """
    seq_id = 0
    print("Start to build CTF data from: {0}".format(corpus))
    with open(corpus, 'r', encoding = 'utf-8') as corp:
      with open(output, 'w', encoding = 'utf-8') as outf:
        all_lines = corp.readlines()
        print("Total lines to prcess: {0}".format(len(all_lines)))
        for line in all_lines:
          line = line.strip('\n')
          ans, query_words, context_words = Vocabulary.parse_corpus_line(line)
          ans_item = entities[ans]
          query_ids = []
          context_ids = []
          is_entity = []
          entity_ids = []
          labels = []
          pos = 0
          answer_idx = None
          for q in context_words:
            if Vocabulary.is_cnn_entity(q):
              item = entities[q]
              context_ids += [ item.id + 1 ]
              entity_ids += [ item.id + 1 ]
              is_entity += [1]
              if ans_item.id == item.id:
                labels += [1]
                answer_idx = pos
              else:
                labels += [0]
            else:
              item = words[q]
              context_ids += [ (item.id + 1 + entities.size) if item != None else 0 ]
              is_entity += [0]
              labels += [0]
            pos += 1
            if (pos >= max_seq_len):
              break
          if answer_idx is None:
            continue
          for q in query_words:
            if Vocabulary.is_cnn_entity(q):
              item = entities[q]
              query_ids += [ item.id + 1 ]
            else:
              item = words[q]
              query_ids += [ (item.id + 1 + entities.size) if item != None else 0 ]
          #Write featurized ids
          outf.write("{0}".format(seq_id))
          for i in range(max(len(context_ids), len(query_ids))):
            if i < len(query_ids):
              outf.write(" |Q {0}:1".format(query_ids[i]))
            if i < len(context_ids):
              outf.write(" |C {0}:1".format(context_ids[i]))
              outf.write(" |E {0}".format(is_entity[i]))
              outf.write(" |L {0}".format(labels[i]))
            if i < len(entity_ids):
              outf.write(" |EID {0}:1".format(entity_ids[i]))
            outf.write("\n")
          seq_id += 1
          if seq_id%1000 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
    print()
    print("Finished to build corpus from {0}".format(corpus))

