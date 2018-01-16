# =============================================================================
# copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk
import numpy as np
import codecs
import os

from cntk.io import UserMinibatchSource, StreamInformation, MinibatchData
from math import ceil, sqrt
from converter import load_vocab_location_from_file, load_vocab_from_file

TEXT_ENCODING = 'utf-8'
UNK = '<unk>'


# a file reader can generate the feature-to-label
class FileReader(object):
    '''A File Reader'''
    def __init__(self, path):
        self.input_file = codecs.open(path, 'r', encoding=TEXT_ENCODING)
        self.pointer = self.generator()
        self.mask = None

    def reset(self):
        self.input_file.seek(0)
        self.pointer = self.generator()
        self.mask = None

    def generator(self):
        '''Get next (feature, label)'''
        for line in self.input_file:
            words = line.split()
            words_count_in_line = len(words)
            for i in range(words_count_in_line):
                if i == words_count_in_line - 1:
                    continue
                yield (words[i], words[i + 1])

    def next(self):
        try:
            if self.mask is not None:
                sample = self.mask
                self.mask = None
            else:
                sample = next(self.pointer)
        except StopIteration:
            return None
        return sample
    
    def hasnext(self):
        if self.mask is not None:
            return True
        else:
            self.mask = self.next()
            if self.mask is not None:
                return True
            else:
                return False


# Provides a override-MinibatchSource for parsing the text to a stream-to-data mapping
class DataSource(UserMinibatchSource):

    def __init__(self, path, word_config, location_config, seqlength, batchsize):
        self.word_index = load_vocab_from_file(word_config)
        self.word_position = load_vocab_location_from_file(location_config)
        self.vocab_dim = len(self.word_index)
        self.vocab_base = int(ceil(sqrt(self.vocab_dim)))
        self.reader = FileReader(path)
        self.seqlength = seqlength
        self.batchsize = batchsize
        
        self.input1 = StreamInformation("input1", 0, 'sparse', np.float32, (self.vocab_base,))
        self.input2 = StreamInformation("input2", 1, 'sparse', np.float32, (self.vocab_base,))
        self.label1 = StreamInformation("label1", 2, 'sparse', np.float32, (self.vocab_base,))
        self.label2 = StreamInformation("label2", 3, 'sparse', np.float32, (self.vocab_base,))
        self.word1 = StreamInformation("word1", 4, 'dense', np.float32, (1,))
        self.word2 = StreamInformation("word2", 5, 'dense', np.float32, (1,))

        super(DataSource, self).__init__()

    def stream_infos(self):
        return [self.input1, self.input2, self.label1, self.label2, self.word1, self.word2]

    def parse_word(self, word):
        # Parse token to id
        return self.word_index[word] if word in self.word_index else self.word_index[UNK]

    def make_minibatch(self, samples):
        # Make the next minibatch
        source = [sample[0] for sample in samples]
        target = [sample[1] for sample in samples]

        def transform(x, w=False):
            return np.reshape(x, (-1, self.seqlength, 1) if w else (-1, self.seqlength))

        source = transform(source)
        target = transform(target)
        input1, label1, input2, label2, word1, word2 = [], [], [], [], [], []
        for i in range(len(source)):
            for w in range(len(source[i])):
                input1.append(self.word_position[source[i][w]][0])
                input2.append(self.word_position[source[i][w]][1])
                label1.append(self.word_position[source[i][w]][1])
                label2.append(self.word_position[target[i][w]][0])
                word1.append(source[i][w])
                word2.append(target[i][w])
        return \
            cntk.Value.one_hot(batch=transform(input1), num_classes=self.vocab_base), \
            cntk.Value.one_hot(batch=transform(input2), num_classes=self.vocab_base), \
            cntk.Value.one_hot(batch=transform(label1), num_classes=self.vocab_base), \
            cntk.Value.one_hot(batch=transform(label2), num_classes=self.vocab_base), \
            cntk.Value(batch=np.asarray(transform(word1, True), dtype=np.float32)), \
            cntk.Value(batch=np.asarray(transform(word2, True), dtype=np.float32))

    def next_minibatch(self, num_samples, number_of_workers=1, worker_rank=0, device=None):
        samples = []
        sweep_end = False
        for i in range(num_samples):
            feature_to_label = self.reader.next()
            if feature_to_label is None:
                samples = samples[: (len(samples) // self.seqlength) * self.seqlength]
                self.reader.reset()
                sweep_end = True
                break
            feature, label = feature_to_label
            curr_word = self.parse_word(feature)
            next_word = self.parse_word(label)
            samples.append((curr_word, next_word))
        batchsize = len(samples) / self.seqlength
        # Divide batch into every gpu
        batchrange = list(map(int, [
                (batchsize // number_of_workers) * worker_rank,
                min((batchsize // number_of_workers) * (worker_rank + 1), batchsize)
            ]))

        samples = samples[batchrange[0] * self.seqlength: batchrange[1] * self.seqlength]
        minibatch = self.make_minibatch(samples)
        sample_count = len(samples)
        num_seq = len(minibatch[0])

        minibatch = {
            self.input1: MinibatchData(minibatch[0], num_seq, sample_count, sweep_end),
            self.input2: MinibatchData(minibatch[1], num_seq, sample_count, sweep_end),
            self.label1: MinibatchData(minibatch[2], num_seq, sample_count, sweep_end),
            self.label2: MinibatchData(minibatch[3], num_seq, sample_count, sweep_end),
            self.word1: MinibatchData(minibatch[4], num_seq, sample_count, sweep_end),
            self.word2: MinibatchData(minibatch[5], num_seq, sample_count, sweep_end)
        }
        return minibatch
