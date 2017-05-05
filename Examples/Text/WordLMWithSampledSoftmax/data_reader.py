# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk as C

# Read the mapping of tokens to ids from a file (tab separated)
def load_token_to_id(token_to_id_file_path):
    token_to_id = {}
    with open(token_to_id_file_path,'r') as f:
       for line in f:
            entry = line.split('\t')
            if len(entry) == 2:
                token_to_id[entry[0]] = int(entry[1])

    return token_to_id

# Provides functionality for reading text file and converting them to mini-batches using a token-to-id mapping from a file.
class DataReader(object):
    def __init__(
        self,
        token_to_id_path,        # file mapping tokens to ids (format: token tab idtokens_per_sequence, sequences_per_minibatch):
        segment_sepparator_token # segment separator
                ):
        self.token_to_id_path = token_to_id_path
        self.token_to_id = load_token_to_id(token_to_id_path)
        self.vocab_dim = len(self.token_to_id)

        if not segment_sepparator_token in self.token_to_id:
            print ("ERROR: separator token '%s' has no id:" % (segment_sepparator_token))
            sys.exit()

        self.segment_sepparator_id = self.token_to_id[segment_sepparator_token]

    # Creates a generator that reads the whole input file and returns mini-batch data as a triple of input_sequences, label_sequences and number of read tokens.
    # Each individual sequence is constructed from one ore more full text lines until the minimal sequence length is reached or surpassed.
    def minibatch_generator(
        self,
        input_text_path,     # Path to text file (train, test or validation data).
        sequence_length,     # Minimal sequence length
        sequences_per_batch, # Number of sequences per batch
                            ):
        with open(input_text_path) as text_file: 
            token_ids = []
            feature_sequences = []
            label_sequences = []
            token_count = 0

            for line in text_file:
                tokens = line.split()

                if len(token_ids) == 0:
                    token_ids.append(self.segment_sepparator_id)

                for token in tokens:
                    if not token in self.token_to_id:
                        print ("ERROR: while reading file '" + input_text_path + "' token without id: " + token)
                        sys.exit()
                    token_ids.append(self.token_to_id[token])

                token_count += len(tokens)

                # When minimum sequence length is reached, create feature and label sequence from it.
                if len(token_ids) >= sequence_length:
                    # We prepend a segment separator before the feature segments
                    feature_sequences.append(token_ids[ : -1])
                    label_sequences.append(token_ids[ 1 :])
                    token_ids = []

                # When the expected number of sequences per batch is reached yield the data and reset the array
                if len(feature_sequences) == sequences_per_batch:
                    yield C.Value.one_hot(feature_sequences, self.vocab_dim), C.Value.one_hot(label_sequences, self.vocab_dim), token_count
                    feature_sequences = []
                    label_sequences   = []
                    token_count = 0

            # From the end of the file there are probably some leftover lines
            if len(feature_sequences) > 0:
                yield C.Value.one_hot(feature_sequences, self.vocab_dim), C.one_hot(label_sequences, self.vocab_dim), token_count



if __name__=='__main__':
    data_reader = DataReader('./ptb/token2id.txt')
    
    print('vocab_dim = ' + str(data_reader.vocab_dim))

    count=0
    for a,b in data_reader.minibatch_generator('./ptb/train.txt', 1, 20):
        count += 1
        print('count:' + str(count))

