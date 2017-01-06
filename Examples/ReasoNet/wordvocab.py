import sys
import os

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

  def push(self, word):
    if word in self.__dict:
      self.__dict[word].freq += 1
    else:
      self.__dict[word] = WordFreq(word, len(self.__dict), 1)

  def build_index(self, max_size):
    items = sorted(self.__dict.values(), key=lambda it : it.freq, reverse=True)
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

  @staticmethod
  def load_bingvocab(vocab_src):
    """
    Load bing vocabulary from file.

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
  def build_bingvocab(input_src, vocab_dst, max_size=50000):
    """
    Build bing vocabulary from raw bing corpus file.

    Args:
      input_src (`str`): the path of the corpus file 
      vocab_dst (`str`): the path of the vocabulary file to save the built vocabulary
      max_size (`int`): the maxium size of the word vocabulary
    Returns:
      :class:`Vocabulary`: Vocabulary of the entities
      :class:`Vocabulary`: Vocabulary of the words
    """
    word_vocab = Vocabulary("WordVocab")
    entity_vocab = Vocabulary("EntityVocab")
    linenum = 0
    with open(input_src, 'r', encoding='utf-8') as src:
      for line in src.readlines():
        ans, query_words, context_words = Vocabulary.parse_bing_corpus_line(line)
        for q in query_words:
          if q.startswith('@'):
            entity_vocab.push(q)
          else:
            word_vocab.push(q)
        for q in context_words:
          if q.startswith('@'):
            entity_vocab.push(q)
          else:
            word_vocab.push(q)
        linenum += 1
        if linenum%1000==0:
          print("{0} lines parsed.".format(linenum))
    entity_vocab.build_index(max_size)
    word_vocab.build_index(max_size)
    with open(vocab_dst, 'w', encoding='utf-8') as dst:
      entity_vocab.save(dst)
      word_vocab.save(dst)
    return entity_vocab, word_vocab

  @staticmethod
  def parse_bing_corpus_line(line):
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

  @staticmethod
  def build_bing_corpus_index(entities, words, corpus, index):
    """
    Build featurized corpus and stored in index file in CNTKTextFormat.

    Args:
      entities (class:`Vocabulary`): The entities vocabulary
      words (class:`Vocabulary`): The words vocabulary
      corpus (`str`): The file path of the raw corpus
      index (`str`): The file path to store the featurized corpus index
    """
    seq_id = 0
    with open(corpus, 'r', encoding = 'utf-8') as corp:
      with open(index, 'w', encoding = 'utf-8') as index:
        for line in corp.readlines():
          ans, query_words, context_words = Vocabulary.parse_bing_corpus_line(line)
          ans_item = entities[ans]
          query_ids = []
          context_ids = []
          is_entity = []
          labels = []
          pos = 0
          for q in context_words:
            if q.startswith('@'):
              item = entities[q]
              context_ids += [ item.id + 1 ]
              is_entity += [1]
              labels += [1 if ans_item.id==item.id else 0]
            else:
              item = words[q]
              context_ids += [ (item.id + 1) if item != None else 0 ]
              is_entity += [0]
              labels += [0]
            pos += 1
          for q in query_words:
            if q.startswith('@'):
              item = entities[q]
              query_ids += [ item.id + 1 ]
            else:
              item = words[q]
              query_ids += [ (item.id + 1) if item != None else 0 ]
          #Write featurized ids
          index.write("{0}".format(seq_id))
          for i in range(max(len(context_ids), len(query_ids))):
            if i < len(query_ids):
              index.write(" |Q {0}:1".format(query_ids[i]))
            if i < len(context_ids):
              index.write(" |C {0}:1".format(context_ids[i]))
              index.write(" |E {0}".format(is_entity[i]))
              index.write(" |L {0}".format(labels[i]))
            index.write("\n")
          seq_id += 1
          if seq_id%1000 == 0:
            print("{0} lines parsed.".format(seq_id))
