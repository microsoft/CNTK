#!/usr/bin/env python

# This script takes a CNTK text format file and a header file, and converts it
# to a CNTK binary format file.
#
# The header file must list all of the streams in the input file in the
# following format:
#   <desired stream name>  <stream alias> <matrix type> <sample dimension>
#
# Where:
#   <desired stream name> is the desired name for the input in CNTK.
#   <stream alias> is the alias for the stream in the input file.
#   <matrix type> is the matrix type, i.e., dense or sparse
#   <sample dimension> is the dimensino of each sample for the input
#

import sys
import argparse
import struct
import os
from collections import OrderedDict

MAGIC_NUMBER = 0x636e746b5f62696e;
CBF_VERSION = 1;

class ElementType:
    FLOAT = 0
    DOUBLE = 1

class MatrixEncodingType:
    DENSE = 0
    SPARSE_CSC = 1
    #COMPRESSED_DENSE = 2
    #COMPRESSED_SPARSE_CSC = 3

# This will convert data in the ctf format into the binary format
class Converter(object):
    def __init__(self, name, sample_dim, element_type):
        self.name = name
        self.sample_dim = sample_dim
        # contains length (in samples) for each sequence in the chunk
        self.sequences = [] 
        self.element_type = element_type

    def write_header(self, output):
        # First is the matrix type.
        output.write(struct.pack('B', self.get_matrix_type()))
        # Nest comes the stream name.
        output.write(struct.pack('I', len(self.name)))
        output.write(self.name.encode('ascii'))
        # Next is the elem type
        output.write(struct.pack('B', self.element_type))
        # Finally, the sample dimension.
        output.write(struct.pack('I', self.sample_dim))

    def write_signed_ints(self, output, ints):
        output.write(b''.join([struct.pack('i', x) for x in ints]))

    def write_floats(self, output, floats):
        format = 'f' if self.is_float() else 'd'
        output.write(b''.join([struct.pack(format, x) for x in floats]))

    def is_float(self):
        return self.element_type == ElementType.FLOAT

    def get_matrix_type(self):
        raise NotImplementedError()

    def reset(self):
        self.sequences = []

    def start_sequence(self):
        self.sequences.append([])

    def add_sample(self, sample):
        raise NotImplementedError()

# Specilization for dense inputs
class DenseConverter(Converter):

    def get_matrix_type(self):
        return MatrixEncodingType.DENSE;

    def add_sample(self, sample):
        if(len(sample) != self.sample_dim):
            raise ValueError(
                "Invalid sample dimension for input {0}".format(self.name))

        byte_size = len(sample) * (4 if self.is_float() else 8)

        if(len(self.sequences) == 0):
            self.sequences.append([])
            byte_size += 4;

        self.sequences[-1].append([float(x) for x in sample])

        return byte_size

    def write_data(self, output):
        for sequence in self.sequences:
            output.write(struct.pack('I', len(sequence)))
            for sample in sequence:
                self.write_floats(output, sample)


# Specialization for sparse inputs
class SparseConverter(Converter):

    def add_sample(self, sample):
        pairs = map(lambda x: (int(x[0]),float(x[1])),
            [pair.split(':', 1) for pair in sample])

        for pair in pairs:
            index = pair[0]
            if (index >= self.sample_dim):
                raise ValueError("Invalid sample dimension for input {0}. Max {1}, given {2}"
                        .format(self.name, self.sample_dim, index))

        byte_size = len(pairs) * (8 if self.is_float() else 12) + 4

        if(len(self.sequences) == 0):
            self.sequences.append([])
            byte_size += 8;

        self.sequences[-1].append(pairs)

        return byte_size

    def get_matrix_type(self):
        return MatrixEncodingType.SPARSE_CSC;

    def write_data(self, output):
        format = 'f' if self.is_float() else 'd'
        for sequence in self.sequences:
            # write out each sequence in csc format
            values = []
            indices = []
            sizes = []
            for sample in sequence:
                sizes.append(len(sample))
                sample.sort(key=lambda x: x[0])
                for (index, value) in sample:
                    indices.append(index)
                    values.append(value)

            output.write(struct.pack('I', len(sequence))) #number of samples in this sequence
            # nnz and indices have to be written out as signed ints, since
            # this is the index type of the CNTK sparse matrix
            output.write(struct.pack('i', len(values))) #total nnz count for this sequence
            self.write_floats(output, values)
            self.write_signed_ints(output, indices)
            self.write_signed_ints(output, sizes)

# Process the entire sequence
def process_sequence(data, converters, chunk):
    byte_size = 0;
    for converter in converters.values():
        converter.start_sequence()
    for line in data:
        for input_stream in line.split("|")[1:]:
            split = input_stream.split(None, 1)
            if (len(split) < 2):
                continue
            (alias, values) = split
            # We need to ignore comments
            if(len(alias) > 0 and alias[0] != '#'):
                byte_size += converters[alias].add_sample(values.split())
    sequence_length_samples = max([len(x.sequences[-1]) for x in converters.values()])
    chunk.add_sequence(sequence_length_samples)
    return byte_size

# Output a binary chunk
def write_chunk(binfile, converters, chunk):
    binfile.flush()
    chunk.offset = binfile.tell()
    # write out the number of samples for each sequence in the chunk
    binfile.write(b''.join([struct.pack('I', x) for x in chunk.sequences]))

    for converter in converters.values():
        converter.write_data(binfile)
        converter.reset()
    # TODO: add a hash of the chunk

def get_converter(input_type, name, sample_dim, element_type):
    if(input_type.lower() == 'dense'):
        return DenseConverter(name, sample_dim, element_type)
    if(input_type.lower() == 'sparse'):
        return SparseConverter(name, sample_dim, element_type)

    raise ValueError('Invalid input format {0}'.format(input_type))

# parse the header to get the converters for this file
# <name>    <alias>  <input format>  <sample size>
def build_converters(header_file, element_type):
    converters = OrderedDict();
    with open(header_file, 'r') as inputs:
        for line in inputs:
            (name, alias, input_type, sample_dim) = line.strip().split()
            converters[alias] = get_converter(input_type, name, int(sample_dim), element_type)
    return converters

class Chunk:
    def __init__(self):
        self.offset = 0
        self.sequences = []

    def num_sequences(self):
        return len(self.sequences)

    def num_samples(self):
        return sum(self.sequences)

    def add_sequence(self, num_samples):
        return self.sequences.append(num_samples)

class Header:
    def __init__(self, converters):
        self.converters = converters
        self.chunks = []

    def add_chunk(self, chunk):
        assert(isinstance(chunk, Chunk))
        self.chunks.append(chunk)

    # Output the binary format header.
    def write(self, output_file):
        output_file.flush()
        header_offset = output_file.tell()
        # First, write the magic number (uint64, 8 bytes)
        output_file.write(struct.pack('Q', MAGIC_NUMBER));
         # Next is the number of chunks (uint32, 4 bytes)
        output_file.write(struct.pack('I', len(self.chunks)))
        # Finally the number of input streams (uint32, 4 bytes)
        output_file.write(struct.pack('I', len(self.converters)))
        for converter in self.converters.values():
            converter.write_header(output_file)
        # write the chunk table
        for chunk in self.chunks:
            # uint64: start offset for chunk
            output_file.write(struct.pack('q', chunk.offset))
            # uint32: number of sequences in the chunk
            output_file.write(struct.pack('I', chunk.num_sequences()))
            # uint32: number of samples in the chunk
            output_file.write(struct.pack('I', chunk.num_samples()))

        output_file.write(struct.pack('q', header_offset));


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transforms a CNTK Text Format file into CNTK binary format given a header.")
    parser.add_argument('--input', help="CNTK Text Format file to convert to binary.", required=True)
    parser.add_argument('--header',  help="Header file describing each stream in the input.", required=True)
    parser.add_argument('--chunk_size', type=int, help='Chunk size in bytes.', required=True)
    parser.add_argument('--output', help='Name of the output file, stdout if not given', required=True)
    parser.add_argument('--precision', help='Floating point precision (double or float). Default is float',
        choices=["float", "double"], default="float", required=False)
    args = parser.parse_args()

    output = open(args.output, "wb")
    # The very first 8 bytes of the file is the CBF magic number.
    output.write(struct.pack('Q', MAGIC_NUMBER));
    # Next 4 bytes is the CBF version.
    output.write(struct.pack('I', CBF_VERSION));

    converters = build_converters(args.header, 
        ElementType.FLOAT if args.precision == 'float' else ElementType.DOUBLE)

    header = Header(converters)
    chunk = Chunk()

    with open(args.input, "r") as input_file:
        sequence = []
        seq_id = None
        estimated_chunk_size = 0
        for line in input_file:
            (prefix, _) = line.rstrip().split('|',1)
            # if the sequence id is empty or not equal to the previous sequence id,
            # we are at a new sequence.
            if((not seq_id and not prefix) or (len(prefix) > 0 and seq_id != prefix)):
                if(len(sequence) > 0):
                    estimated_chunk_size += process_sequence(sequence, converters, chunk)
                    sequence = []
                    if(estimated_chunk_size >= int(args.chunk_size)):
                        write_chunk(output, converters, chunk)
                        header.add_chunk(chunk)
                        chunk = Chunk()
                seq_id = prefix

            sequence.append(line)
        # we must parse the last line
        if(len(sequence) > 0):
            process_sequence(sequence, converters, chunk)

        write_chunk(output, converters, chunk)
        header.add_chunk(chunk)

        header.write(output)

        output.flush()
        output.close()