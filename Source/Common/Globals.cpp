//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Globals.h"
#include <unordered_map>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    // TODO: get rid of this source file once static initializers in methods are thread-safe (VS 2015)
    std::atomic<bool> Globals::m_forceDeterministicAlgorithms(false);
    std::atomic<bool> Globals::m_forceConstantRandomSeed(false);

    std::atomic<bool> Globals::m_enableShareNodeValueMatrices(true);
    std::atomic<bool> Globals::m_optimizeGradientAccumulation(true);

    // Note: this is a map that transfers the old reader and writer names to
    //       the new naming scheme
    std::unordered_map<std::wstring, std::wstring> g_deprecatedReaderWriterNameMap =
    {
        // legacy reader mapping
        { L"HTKMLFReader",          L"Cntk.Reader.HTKMLF" },
        { L"LMSequenceReader",      L"Cntk.Reader.LMSequence" },
        { L"LUSequenceReader",      L"Cntk.Reader.LUSequence" },
        { L"UCIFastReader",         L"Cntk.Reader.UCIFast" },
        { L"LibSVMBinaryReader",    L"Cntk.Reader.SVMBinary" },
        { L"SparsePCReader",        L"Cntk.Reader.SparsePC" },
        { L"Kaldi2Reader",          L"Cntk.Reader.Kaldi2" },
        { L"BinaryReader",          L"Cntk.Reader.Binary" },

        // legacy writer mapping
        { L"HTKMLFWriter",          L"Cntk.Reader.HTKMLF" },
        { L"BinaryWriter",          L"Cntk.Reader.Binary" },
        { L"LUSequenceWriter",      L"Cntk.Reader.LUSequence" },
        { L"LMSequenceWriter",      L"Cntk.Reader.LMSequence" },
        { L"Kaldi2Writer",          L"Cntk.Reader.Kaldi2" },

        // New type of readers/writers
        { L"CompositeDataReader",   L"Cntk.Composite" },
        { L"HTKDeserializers",      L"Cntk.Deserializers.HTK" },
        { L"CNTKTextFormatReader",  L"Cntk.Deserializers.TextFormat" },
        { L"CNTKBinaryReader",      L"Cntk.Deserializers.Binary" },
        { L"ImageReader",           L"Cntk.Deserializers.Image" },
    };

}}}
