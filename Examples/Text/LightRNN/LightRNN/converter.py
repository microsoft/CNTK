# =============================================================================
# copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import division
from math import ceil, floor, sqrt
from random import shuffle

import glob
import operator
import codecs

TEXT_ENCODING = 'utf-8'
UNK = '<unk>'


def load_vocab_from_file(vocab_file):
    # Load vocabulary table from file
    vocab = {}
    with codecs.open(vocab_file, 'r', encoding=TEXT_ENCODING) as input_file:
        for line in input_file:
            word = line.strip()
            vocab[word] = len(vocab)
    return vocab


def load_vocab_location_from_file(location_file):
    # Load vocabulary table location from file
    location = {}
    with codecs.open(location_file, 'r', encoding=TEXT_ENCODING) as input_file:
        row_id = 0
        for row in input_file:
            cols = list(map(int, row.split()))
            for col_id in range(len(cols)):
                if cols[col_id] != -1:
                    location[cols[col_id]] = (row_id, col_id)
            row_id += 1
    return location


def read_line(line, word_count):
    # Count word and frequency by one line
    words = line.split()
    for word in words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1


def generate_vocab_from_source_file(input_dir, freq=None):
    # Generate the vocabulary table from the files of the input dir
    files = glob.glob(input_dir + '/*.txt')
    word_index = {}
    word_count = {}
    for file in files:
        with codecs.open(file, 'r', encoding=TEXT_ENCODING) as input_file:
            for line in input_file:
                read_line(line, word_count)
    print ('Get {} words from {} files'.format(len(word_count), len(files)))
    if freq:
        print ('Get Top {} words by frequency'.format(freq))
        ordered_words_by_index = sorted(word_count.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)
        ordered_words_by_index = ordered_words_by_index[:freq]
        for word in ordered_words_by_index:
            word_index[word[0]] = len(word_index)
            if len(word_index) == freq and UNK not in word_index:
                del word_index[word[0]]
                word_index[UNK] = len(word_index)
    else:
        for word in word_count:
            word_index[word] = len(word_index)
    return word_index


def init_location(word_index):
    # Init the word table location
    vocab_size = len(word_index)
    base_size = ceil(sqrt(vocab_size))
    random_locations = [i for i in range(vocab_size)]
    shuffle(random_locations)
    word_location = {}
    for word in word_index:
        i = word_index[word]
        random_index = random_locations[i]
        word_location[i] = (floor(random_index / base_size),
                            random_index % base_size)
    return word_location


def save_to_vocab_file(word_index, vocab_file):
    # save the word table to file
    with codecs.open(vocab_file, 'w', encoding=TEXT_ENCODING) as input_file:
        for word in word_index:
            input_file.write(word + '\n')


def save_to_vocab_location_file(word_location, vocab_location_file):
    # save the word location table to file
    vocab_size = len(word_location)
    sqrt_length = int(ceil(sqrt(vocab_size)))
    locations = {}
    for i in range(sqrt_length):
        for j in range(sqrt_length):
            locations[i, j] = -1
    for word, location in word_location.items():
        locations[location[0], location[1]] = word

    with codecs.open(vocab_location_file, 'w', encoding=TEXT_ENCODING) as output_file:
        for i in range(sqrt_length):
            for j in range(sqrt_length):
                output_file.write('%d ' % locations[i, j])
            output_file.write('\n')
    print ('save %d location into %s' % (vocab_size, vocab_location_file))
