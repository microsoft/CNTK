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

    // TODO: this is a map that transfers the old reader and writer names to
    //       the new naming scheme
    std::unordered_map<std::wstring, std::wstring> s_deprecatedReaderWriterNameMap =
    {
        // reader mapping
        { L"HTKMLFReader", L"Cntk.Reader.HTKMLF" },
        { L"CompositeDataReader", L"Cntk.Reader.CompositeData" },
        { L"HTKDeserializers", L"Cntk.Reader.HTKDeserializer" },
        { L"LMSequenceReader", L"Cntk.Reader.LMSequence" },
        { L"LUSequenceReader", L"Cntk.Reader.LUSequence" },
        { L"UCIFastReader", L"Cntk.Reader.UCIFast" },
        { L"LibSVMBinaryReader", L"Cntk.Reader.SVMBinary" },
        { L"SparsePCReader", L"Cntk.Reader.SparsePC" },
        { L"CNTKBinaryReader", L"Cntk.Reader.Binary" },
        { L"CNTKTextFormatReader", L"Cntk.Reader.TextFormat" },
        { L"Kaldi2Reader", L"Cntk.Reader.Kaldi" },
        { L"ImageReader", L"Cntk.Reader.Image" },
        { L"BinaryReader", L"Cntk.Reader.Binary.Deprecated" },

        // writer mapping
        { L"HTKMLFWriter", L"Cntk.Reader.HTKMLF" },
        { L"BinaryWriter", L"Cntk.Reader.Binary.Deprecated" },
        { L"LUSequenceWriter", L"Cntk.Reader.LUSequence" },
        { L"LMSequenceWriter", L"Cntk.Reader.LMSequence" },
        { L"Kaldi2Writer", L"Cntk.Reader.Kaldi" },
    };

}}}
