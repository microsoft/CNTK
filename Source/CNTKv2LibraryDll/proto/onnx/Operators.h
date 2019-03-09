//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

#include <set>
#include <regex>

namespace onnxruntime
{
class Graph;
}

// ONNX reshape spec: In this case, the value is inferred from the size of the tensor and the remaining dimensions
const int64_t ReshapeInferredDim = -1;
// ONNX reshape spec: the actual dimension value is unchanged(i.e.taken from the input tensor).
const int64_t ReshapeKeepInputDim = 0;
const std::string FreeSequenceDimParam = "Sequence";
const size_t numBiasInOnnxLstm = 2; // bias for W, and bias for R (also called H in CNTK).
                                    // TODO: support cases where batch size is not 1.

// See comment for BatchSizeOverride
// we need to object to keep OverridedBatch dimension in case defaultFreeBatchSize is overrided 
// by broadcast (and possible other ops TO BE FIGURED OUT) ops. 
class BatchSizeProcessor
{
public:
    static int FreeBatchSize()
    {
        return overrideBatchSize;
    }

    static void OverrideBatchSize(int i_overrideBatchSize)
    {
        // TODO: this does not work completely. 
        // TODO: Waiting Skype smart reply with attention model before enabling the functionality of tracking sequence dimension.
        // overrideBatchSize = i_overrideBatchSize;
    }

    static void ResetOverrideBatchSize()
    {
        overrideBatchSize = defaultFreeBatchSize;
    }

    static size_t FreeSequenceSize()
    {
        return overrideSequenceSize;
    }
    static void OverrideSequenceSize(size_t i_overrideSequenceSize)
    {
        overrideSequenceSize = i_overrideSequenceSize;
    }

    static void ResetOverrideSequenceSize()
    {
        overrideSequenceSize = CNTK::NDShape::FreeDimension;
    }
private:
    static const int defaultFreeBatchSize = 1;
    static int overrideBatchSize;

    static size_t overrideSequenceSize;
};

namespace CNTK
{
namespace ONNX
{
struct AttributesMapping
{
    std::unordered_map<std::wstring, std::string> map;
};

class Operators
{
public:
    //
    // Check if opName is one of the supported ONNX OP.
    //
    static inline bool IsSupportedCNTKOP(const std::wstring& opName)
    {
        return _cntkToONNXOpName.find(opName) != _cntkToONNXOpName.end();
    }

    static inline bool IsBlockFnNotConvertedThroughBlockRoot(FunctionPtr blkF)
    {
        return 
            blkF->OpName() == L"Sequence::BroadcastAs" || 
            blkF->OpName() == L"ElementMax" || 
            blkF->OpName() == L"Convolution";
    }

    //
    // Layer APIs use block function as a wrapper, so we need to handle them with care.
    //
    static inline bool IsLayerCNTKOP(const std::wstring& opName)
    {
        return _cntkLayerOPName.find(opName) != _cntkLayerOPName.end();
    }

    //
    // Return a lookup table which is keyed on CNTK OP, and the value is another table
    // that contain name mapping from CNTK to ONNX.
    //
    static inline const std::unordered_multimap<std::wstring, AttributesMapping>& CntkToONNXLookup()
    {
        return _cntkToONNXOpName;
    }

    //
    // Method to check if a name is a valid optimizedRnnStack op name.
    //
    static inline bool IsOptimizedRnnStackOp(const std::wstring& opName)
    {
        return _optimizedRnnStackOpNames.find(opName) != _optimizedRnnStackOpNames.end();
    }

    //
    // Return a lookup table that maps CNTK's optimizedRNNStack op to one of the
    // ONNX RNN ops.
    //
    static inline const std::unordered_map<std::wstring, std::string>& OptimizedRnnToOnnxOpLookup()
    {
        return _optimizedRnnOpNameToOnnxOpName;
    }

    //
    // Check if this CNTK op corresponds to an ONNX op that has a defined batch axis.
    //
    static inline bool IsOpExportedWithBatchAxis(const std::wstring& opName)
    {
        return _cntkOpsExportedWithBatchAxis.find(opName) != _cntkOpsExportedWithBatchAxis.end();
    }

    //
    // Check if the ONNX op is a simple op with batch axis.
    //
    static inline bool IsSimpleBatchAxisOnnxOp(const std::string& opName)
    {
        return _onnxSimpleBatchAxisOps.find(opName) != _onnxSimpleBatchAxisOps.end();
    }


    static std::tuple<int, int> GetElementWiseInputIndices(const std::wstring& opName);

    //
    // Because in CNTK block, we can't filtered out the external inputs to the block.
    // We need a way to filter out leaf input from its subgraph.
    //
    static inline bool IsValidInputs(const std::wstring& opName, size_t index)
    {
        if (_cntkBlockOPInvalidIndices.find(opName) == _cntkBlockOPInvalidIndices.end())
            return true;

        auto invalidIndices = _cntkBlockOPInvalidIndices[opName];
        return invalidIndices.find(index) == invalidIndices.end();
    }

    //
    // The positional of the argument between CNTK and ONNX aren't the same.
    // The below function return true, if we need a remap.
    //
    static inline bool HasInputIndexMap(const std::wstring& opName)
    {
        return _cntkToONNXInputIndices.find(opName) != _cntkToONNXInputIndices.end();
    }

    //
    // If we need a remap, the below function return a remapping map.
    //
    static inline const std::vector<int>& ToONNXInputIndexMap(const std::wstring& opName)
    {
        assert(_cntkToONNXInputIndices.find(opName) != _cntkToONNXInputIndices.end());
        return _cntkToONNXInputIndices[opName];
    }

    //
    // For block function with internal constant or parameter, we don't want to create
    // the corresponding ONNX tensor for some of the parameters.
    //
    static inline bool IgnoreConstantAndParameter(const std::wstring& opName, size_t index)
    {
        if (_cntkToONNXInputIndices.find(opName) != _cntkToONNXInputIndices.end())
        {
            auto indexMap = _cntkToONNXInputIndices[opName];
            assert(index < indexMap.size());

            return (indexMap[index] < 0);
        }

        return false;
    }

    static const AttributesMapping& FindAttributeMap(const std::wstring& cntkOpName, const std::wstring& cntkAttributeOpName);

    static bool SupportBroadcast(const std::wstring& cntkOpName);
    static bool SupportBroadcastONNXOp(const std::string& onnxOpName);

    static bool IsLoopOp(const std::string &opName);
    static bool IsRNNOp(const std::string &opName);
    static bool IsSequenceBlockOp(const std::string &opName);

private:
    static std::unordered_multimap<std::wstring, AttributesMapping> _cntkToONNXOpName;
    static std::unordered_map<std::wstring, std::set<size_t>> _cntkBlockOPInvalidIndices;
    static std::unordered_map<std::wstring, std::vector<int>> _cntkToONNXInputIndices;
    static std::set<std::wstring>_optimizedRnnStackOpNames;
    static std::unordered_map<std::wstring, std::string> _optimizedRnnOpNameToOnnxOpName;
    static std::set<std::wstring> _cntkLayerOPName;
    static std::set<std::wstring> _cntkOpsExportedWithBatchAxis;
    static std::set<std::string> _onnxSimpleBatchAxisOps;
};

std::string GetRootPath(const std::string& rootPath);
} // namespace ONNX
} // namespace CNTK
