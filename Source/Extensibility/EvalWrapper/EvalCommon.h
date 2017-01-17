//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalCommon.h -- Common structures used by managed code wrapping the native EvaluateModel interface
//

namespace Microsoft { namespace MSR { namespace CNTK { namespace Extensibility { namespace Managed {

/// Enumeration for the types of nodes
public enum class NodeGroup
{
    Input,  // an input node
    Output, // an output node
    Specified
};

public enum class DataType
{
    Float32,
    Float64
};

public enum class StorageType
{
    Unknown,
    Dense,
    Sparse,
};

}}}}}