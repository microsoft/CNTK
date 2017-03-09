#!/usr/bin/env python

# This script takes a CNTK text format file and a header file, and converts it
# to a CNTK binary format file.
#
# The header file must list all of the streams in the input file in the
# following format:
#   <desired stream name> TAB <stream alias> TAB <matrix type> TAB <sample dimension>
#
# Where:
#   <desired stream name> is the desired name for the input in CNTK.
#   <stream alias> is the alias for the stream in the input file.
#   <matrix type> is the matrix type, i.e., dense or sparse
#   <sample dimension> is the dimensino of each sample for the input
#

import sys
import argparse
import re
import struct
import tempfile
import shutil
import os

# This will convert data in the ctf format into binary format
class Converter(object):
    def __init__(self, name, sampleDim):
        self.name = name
        self.sampleDim = sampleDim
        self.vals = list()

    def getName(self):
        return self.name

    def getSampleDim(self):
        return self.sampleDim

    def clear(self):
        self.vals = list()

    def addSequence(self):
        self.vals.append(list())

    def appendSample(self, sample):
        if( len(sample) != self.sampleDim ):
            print( "Invalid sample dimension for input {0}" ).format( self.name )
            sys.exit()
        if( len(self.vals) == 0 ):
            self.vals.append( list() )

        self.vals[-1].append( sample )

    def toString(self):
        output = ""
        for seq in self.vals:
            for samp in seq:
                output += "\t" + " ".join(samp )
            output += "\n"
        return output


# Specilization for dense inputs
class DenseConverter(Converter):
    def __init__(self, name, sampleDim):
        Converter.__init__(self, name, sampleDim)
    
    def headerBytes(self):
        output = ""
        # First is the matrix type. Dense is type 0
        output += struct.pack( "i", 0 )
        # Next is the elem type, currently float only
        output += struct.pack( "i", 0 )
        # Finally is whether or not this is a sequence
        output += struct.pack( "i", self.sampleDim )

        return output

    
    def toBytes(self):
        output = ""
        for sequence in self.vals:
            if( len(sequence) != 1 ):
                print( "Converter does not support dense sequences." )
                sys.exit()
            for sample in sequence[0]:
                output += struct.pack( "f", float(sample) )

        return output


# Specialization for sparse inputs
class SparseConverter(Converter):
    def __init__(self, name, sampleDim):
        Converter.__init__(self, name, sampleDim)

    def appendSample(self, sample):
        for samp in sample:
            if( int(samp.split(":")[0]) >= self.sampleDim ):
                print( "Invalid sample dimension for input {0}. Max {1}, given {2}" ).format( self.name, self.sampleDim, sample.split( ":" )[0] )
                sys.exit()
        if( len(self.vals) == 0 ):
            self.vals.append( list() )

        self.vals[-1].append( sample )

    def headerBytes(self):
        output = ""
        # First is the matrix type. Sparse is type 1
        output += struct.pack( "i", 1 )
        # Next is the storage type, currently sparse csc only
        output += struct.pack( "i", 0 )
        # Next is the elem type, currently float only
        output += struct.pack( "i", 0 )
        # Next is whether or not this is a sequence
        # Note this is currently ignored
        output += struct.pack( "i", 1 )
        # Finally is the sample dimension
        output += struct.pack( "i", self.sampleDim )

        return output

    def toBytes(self):
        output = ""
        values = list()
        rowInd = list()
        colInd = [0]
        nnz = 0
        for sequence in self.vals:
            i = 0
            for sample in sequence:
                # sort the indices least to greatest
                sample.sort(key=lambda x: int(x.split(":")[0]))
                for ele in sample:
                    nnz += 1
                    ind, val = ele.split(":")
                    rowInd.append( int(ind) + i * self.sampleDim )
                    values.append( val ) 
                i += 1
            colInd.append( nnz )

        output += struct.pack( "i", nnz )
        output += "".join( [ struct.pack( "f", float(val) ) for val in values ] )
        output += "".join( [ struct.pack( "i", int(ind) ) for ind in rowInd ] )
        output += "".join( [ struct.pack( "i", int(ind) ) for ind in colInd ] )

        return output


# Parse an entire sequence given an aliasToId map, and the converters
def ParseSequence( aliasToId, curSequence, converters ):
    for des in converters:
        des.addSequence()
    for line in curSequence:
        for input in line.split( "|" )[1:]:
            vals = input.split()
            # We need to ignore comments
            if( vals[0] != "#" ):
                converters[aliasToId[vals[0]]].appendSample( vals[1:] )
    return max( [ len(des.vals[ -1 ]) for des in converters ] )

# Output a binary chunk
def OutputChunk( binfile, converters ):
    startPos = binfile.tell()
    for des in converters:
        binfile.write( des.toBytes() )
        des.clear()
    return startPos

# Get a converter from a type
def GetConverter( inputtype, name, sampleDim ):
    converter = None
    if( inputtype.lower() == 'dense' ):
        converter = DenseConverter( name, sampleDim )
    elif( inputtype.lower() == 'sparse' ):
        converter = SparseConverter( name, sampleDim )
    else:
        print( 'Invalid input format {0}' ).format( inputtype )
        sys.exit()

    return converter 

# Output the binary format header.
def OutputHeader( headerFile, converters ):
    # First the version number
    headerFile.write( struct.pack( "q", 1 ) )
    # Next is the number of chunks, but we don't know what this is, so write a
    # placeholder
    headerFile.write( struct.pack( "q", 0 ) )
    # Finally the number of inputs
    headerFile.write( struct.pack( "i", len(converters) ) )
    for conv in converters:
        # first comes the name. This is common so write it first
        headerFile.write( struct.pack( "i", len( conv.getName() ) ) )
        headerFile.write( conv.getName().encode('ascii') )
        headerFile.write( conv.headerBytes() )
        
        
# At the end we know how many chunks there are. Update the header as needed.
def UpdateHeader( headerFile, numChunks ):
    # seek after the first Int64
    headerFile.seek( 8 )
    # Write the number of chunks
    headerFile.write( struct.pack( "q", numChunks ) )

# Output a single row of the offsets table
def OutputOffset( headerFile, numBytes, numSeqs, numSamples ):
    # Int64 start offset for chunk
    headerFile.write( struct.pack( "q", numBytes ) )
    # Int32 Num sequences in the chunk
    headerFile.write( struct.pack( "i", numSeqs ) )
    # Int32 Num samples in the chunk
    headerFile.write( struct.pack( "i", numSamples ) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transforms a CNTK Text Format file into CNTK binary format given a header.")
    parser.add_argument('--input', help="CNTK Text Format file to convert to binary.", default="", required=True)
    parser.add_argument('--header',  help="Header file describing each stream in the input.", default="", required=True)
    parser.add_argument('--seqsPerChunk', type=int, help='Number of sequences in each chunk.', default="", required=True)
    parser.add_argument('--output', help='Name of the output file, stdout if not given', default="", required=True)
    args = parser.parse_args()

    # Since we don't know how many chunks we're going to write until we're done,
    # grow the header/offsets table and the data portion separately. then at the
    # end concatenate the data portion onto the end of the header/offsets
    # portion.
    binaryHeaderFile = open( args.output, "wb+" )
    binaryDataFile = tempfile.NamedTemporaryFile(mode="rb+", delete=False)
    dataPath = binaryDataFile.name

    # parse the header to get the converters for this file
    # <name>    <alias>  <input format>  <sample size>
    converters = []
    aliasToId = dict()
    with open( args.header, "r" ) as headerfile:
        id = 0
        for line in headerfile:
            split = re.split(r'\t+', line.strip())
            converters.append( GetConverter( split[ 2 ], split[ 0 ], int(split[3]) ) )
            aliasToId[ split[ 1 ] ] = id
            id += 1

    OutputHeader( binaryHeaderFile, converters )

    numChunks = 0
    with open( args.input, "r" ) as inputFile:
        curSequence = list()
        numSeqs = 0
        numSamps = 0
        prevId = None
        for line in inputFile:
            split = line.rstrip().split('|')
            # if the sequence id is empty or not equal to the previous sequence id,
            # we are at a new sequence.
            if( not split[0] or prevId != split[ 0 ] ):
                if(len(curSequence) > 0):
                    numSamps += ParseSequence( aliasToId, curSequence, converters )
                    curSequence = list()
                    numSeqs += 1
                    if( numSeqs % int( args.seqsPerChunk ) == 0 ):
                        numBytes = OutputChunk( binaryDataFile, converters )
                        numChunks += 1
                        OutputOffset( binaryHeaderFile, numBytes, numSeqs, numSamps )
                        numSeqs = 0
                        numSamps = 0
                prevId = split[ 0 ]

            curSequence.append( line )
        # we must parse the last line
        if( len(curSequence) > 0 ):
            numSamps += ParseSequence( aliasToId, curSequence, converters )
            numSeqs += 1
            numChunks += 1

        numBytes = OutputChunk( binaryDataFile, converters )
        OutputOffset( binaryHeaderFile, numBytes, numSeqs, numSamps )

        UpdateHeader( binaryHeaderFile, numChunks )
        binaryHeaderFile.flush()
        binaryDataFile.flush()
        binaryHeaderFile.close()
        binaryDataFile.close()

        destination = open( args.output, 'awb+' )
        shutil.copyfileobj( open( dataPath, "rb" ), destination )
        
        destination.flush()
        destination.close()
        os.unlink(dataPath)
        assert not os.path.exists(dataPath)
