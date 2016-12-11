using System;
using System.Collections.Generic;

namespace CNTK
{
    public sealed class SequenceOneHotVector : List<uint>
    {
        public SequenceOneHotVector(uint vocabSize)
        {
            VocabSize = vocabSize;
        }

        public uint VocabSize { get; private set; }
    }
}
