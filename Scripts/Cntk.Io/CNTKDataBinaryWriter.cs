using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Cntk.Io
{
    /// <summary>Chunked columnar data storage.</summary>
    /// <remarks>
    /// The CNTK binary format is columnar. Thus this writer
    /// is designed to take advantage of this columnar design
    /// from a performance perspective. In particular, if the
    /// original data is already columnar, the data can be streamed
    /// in a columnar fashion to the writer.
    /// 
    /// The amount of memory required by the writer is proportional
    /// to the size of the largest chunk. Finalizing a chunk frees
    /// the memory associated with the chunk.
    /// 
    /// See also
    /// https://docs.microsoft.com/en-us/cognitive-toolkit/BrainScript-CNTKBinary-Reader
    /// </remarks>
    public class CNTKDataBinaryWriter
    {
        /// <summary>Spells as 'cntk_bin' reversed</summary>
        const ulong MagicNumber = 0x636e746b5f62696e;
        const int CbfVersion = 1;

        class Chunk
        {
            /// <summary>One writer per input stream.</summary>
            private readonly BinaryWriter[] _writers;

            /// <summary>Intended for chunk statistics (part of the binary format).</summary>
            private int _sampleCount;


            /// <summary>Defensive helper.</summary>
            private static void ThrowIfNaN(float value)
            {
                if (float.IsNaN(value))
                {
                    throw new ArgumentException("Can't serialize NaN.");
                }
            }

            public Chunk(int inputStreamCount)
            {
                _writers = new BinaryWriter[inputStreamCount];
                for (var i = 0; i < inputStreamCount; i++)
                {
                    _writers[i] = new BinaryWriter(new MemoryStream());
                }

                _sampleCount = 0;
            }

            public void WriteDense(int streamIndex, float value)
            {
                ThrowIfNaN(value);

                var writer = _writers[streamIndex];

                writer.Write(1);
                writer.Write(value);
                _sampleCount += 1;
            }

            public void WriteDense(int streamIndex, float[] values, int offset, int count)
            {
                var writer = _writers[streamIndex];

                writer.Write(count);
                for (var i = offset; i < offset + count; i++)
                {
                    ThrowIfNaN(values[i]);
                    writer.Write(values[i]);
                }

                _sampleCount += count;
            }

            public void WriteSparse(int streamIndex, int rowIndex, float value)
            {
                ThrowIfNaN(value);

                var writer = _writers[streamIndex];

                writer.Write(1); // number of sub-sequences (numSamples)
                writer.Write(1); // number of non-zero values (nnz)
                writer.Write(value);
                writer.Write(rowIndex);
                writer.Write(1); // condensed column index

                _sampleCount += 1;
            }

            public void WriteSparse(int streamIndex, Sample[] sortedSamples, int offset, int count)
            {
                var writer = _writers[streamIndex];

                writer.Write(1); // number of sub-sequences (numSamples)
                writer.Write(count); // number of non-zero values (nnz)
                for (var i = offset; i < offset + count; i++)
                {
                    ThrowIfNaN(sortedSamples[i].Value);
                    writer.Write(sortedSamples[i].Value);
                }
                for (var i = offset; i < offset + count; i++)
                {
                    writer.Write(sortedSamples[i].Index);
                }
                // odd but reflect our lack of support for sub-sequences
                writer.Write(count); // number of non-zero values (nnz) per sub-sequences

                _sampleCount += count;
            }

            public void WriteSparse(int streamIndex, int[] sortedRowIndices, float[] values)
            {
                var writer = _writers[streamIndex];

                writer.Write(1); // number of sub-sequences (numSamples)
                writer.Write(sortedRowIndices.Length); // number of non-zero values (nnz)
                for (var i = 0; i < sortedRowIndices.Length; i++)
                {
                    ThrowIfNaN(values[i]);
                    writer.Write(values[i]);
                }
                for (var i = 0; i < sortedRowIndices.Length; i++)
                {
                    writer.Write(sortedRowIndices[i]);
                }
                // odd but reflect our lack of support for sub-sequences
                writer.Write(sortedRowIndices.Length); // number of non-zero values (nnz) per sub-sequences

                _sampleCount += sortedRowIndices.Length;
            }

            public ChunkStat FinalizeAndClear(BinaryWriter output, int sequenceCount)
            {
                output.Flush();
                var offset = output.BaseStream.Position;

                // The number of samples within each sequence
                for (var i = 0; i < sequenceCount; i++)
                {
                    // The reason to have them there is exactly that it allows to 
                    // retrieve all metadata required by the reader (i.e., the number 
                    // of samples for each sequence in a chunk, number of sequences 
                    // in each chunk) without having to parse the input. This is one 
                    // of the constraints imposed by the current randomization 
                    // implementation -- to create a schedule of when each chunk 
                    // should be loaded, it first needs to construct a global timeline, 
                    // which requires all this meta-info. It's possible to omit the 
                    // number of samples when writing the actual sequence data, but 
                    // the amount of space saved this way is too negligible to 
                    // justify the bookkeeping overhead.

                    // HACK: we are approximating the number of samples here
                    output.Write(_writers.Length);
                }

                // columnar data layout inside the chunk
                for (var i = 0; i < _writers.Length; i++)
                {
                    var writer = _writers[i];
                    var stream = (MemoryStream)writer.BaseStream;
                    stream.Position = 0;
                    stream.CopyTo(output.BaseStream);
                }

                // collect the stats
                var stat = new ChunkStat(offset,
                    /* num_sequences */ sequenceCount,
                    /* num_samples*/ _sampleCount);

                // clear the chunk
                for (var i = 0; i < _writers.Length; i++)
                {
                    var writer = _writers[i];
                    var stream = (MemoryStream)writer.BaseStream;
                    stream.SetLength(0);
                }
                _sampleCount = 0;

                return stat;
            }
        }

        class ChunkStat
        {
            public long Offset { get; }

            public int NumSequences { get; }

            public int NumSamples { get; }

            public ChunkStat(long offset, int numSequences, int numSamples)
            {
                Offset = offset;
                NumSequences = numSequences;
                NumSamples = numSamples;
            }
        }

        private readonly BinaryWriter _writer;

        private readonly Chunk _currentChunk;

        private readonly List<ChunkStat> _chunkStats;

        /// <param name="output">The output of this writer.</param>
        /// <param name="inputStreamCount">The number of CNTK input streams 
        /// (CNTK jargon, not to be confused with .NET streams).</param>
        public CNTKDataBinaryWriter(Stream output, int inputStreamCount)
        {
            _writer = new BinaryWriter(output, Encoding.ASCII, true);
            _currentChunk = new Chunk(inputStreamCount);

            _writer.Write(MagicNumber);
            _writer.Write(CbfVersion);
            _chunkStats = new List<ChunkStat>();
        }

        public void WriteDense(int streamIndex, float value)
        {
            _currentChunk.WriteDense(streamIndex, value);
        }

        public void WriteDense(int streamIndex, float[] values)
        {
            _currentChunk.WriteDense(streamIndex, values, 0, values.Length);
        }

        /// <summary>Write a segment of the input array as dense values.</summary>
        /// <remarks>The 'offset' and 'count' arguments offer the possibility to pool arrays.</remarks>
        public void WriteDense(int streamIndex, float[] values, int offset, int count)
        {
            _currentChunk.WriteDense(streamIndex, values, offset, count);
        }

        public void WriteSparse(int streamIndex, int rowIndex, float value)
        {
            _currentChunk.WriteSparse(streamIndex, rowIndex, value);
        }

        public void WriteSparse(int streamIndex, Sample sample)
        {
            _currentChunk.WriteSparse(streamIndex, sample.Index, sample.Value);
        }

        public void WriteSparse(int streamIndex, int[] sortedRowIndices, float[] values)
        {
            _currentChunk.WriteSparse(streamIndex, sortedRowIndices, values);
        }

        public void WriteSparse(int streamIndex, Sample[] sortedSamples)
        {
            _currentChunk.WriteSparse(streamIndex, sortedSamples, 0, sortedSamples.Length);
        }

        /// <summary>Write a segment of the input array as dense values.</summary>
        /// <remarks>The 'offset' and 'count' arguments offer the possibility to pool arrays.</remarks>
        public void WriteSparse(int streamIndex, Sample[] sortedSamples, int offset, int count)
        {
            _currentChunk.WriteSparse(streamIndex, sortedSamples, offset, count);
        }

        public void FinalizeChunk(int sequenceCount)
        {
            _chunkStats.Add(_currentChunk.FinalizeAndClear(_writer, sequenceCount));
        }

        /// <summary>Headers are appended at the end of the file.</summary>
        public void FinalizeFile(CNTKMatrixHeader[] headers)
        {
            _writer.Flush();
            var headerOffset = _writer.BaseStream.Position;

            _writer.Write(MagicNumber);
            _writer.Write(_chunkStats.Count); // number of chunks

            _writer.Write((int)headers.Length); // number of streams

            // stream headers
            foreach (var header in headers)
            {
                _writer.Write((byte)header.MatrixEncodingType);

                _writer.Write(header.Name.Length);
                foreach (var c in header.Name)
                {
                    _writer.Write(c);
                }

                _writer.Write((byte)header.ElementType);

                _writer.Write(header.SampleDimension);
            }

            // chunk headers
            foreach (var stat in _chunkStats)
            {
                _writer.Write(stat.Offset);
                _writer.Write(stat.NumSequences);
                _writer.Write(stat.NumSamples);
            }

            // ZIP-style offset
            _writer.Write(headerOffset);
        }
    }
}
