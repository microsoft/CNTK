//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

#include <set>

namespace ONNXIR
{
    class Graph;
}

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
            // Because in CNTK block, we can't filtered out the external inputs to the block.
            // We need a way to filter out leaf input from its subgraph.
            //
            static inline bool IsValidInputs(const std::wstring& opName, size_t index)
            {
                assert(_cntkBlockOPInvalidIndices.find(opName) != _cntkBlockOPInvalidIndices.end());

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

        private:
            static std::unordered_multimap<std::wstring, AttributesMapping> _cntkToONNXOpName;
            static std::unordered_map<std::wstring, std::set<size_t>> _cntkBlockOPInvalidIndices;
            static std::unordered_map<std::wstring, std::vector<int>> _cntkToONNXInputIndices;
            static std::set<std::wstring> _cntkLayerOPName;
        };

    }
}