//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKException.h -- Managed CNTK Exception wrappers
//

#include "ExceptionWithCallStack.h"

using namespace std;
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;
using namespace System::Runtime::Serialization;
using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Extensibility { namespace Managed {

[Serializable]
public ref class CNTKException : Exception, ISerializable
{
public:
    CNTKException() : Exception()
    {}

    CNTKException(String^ message) : Exception(message)
    {}

    CNTKException(String^ message, String^ callstack) : Exception(message), NativeCallStack(callstack)
    {}

    const String^ NativeCallStack;

protected:

    CNTKException(SerializationInfo^ info, StreamingContext context) : Exception(info, context)
    {}
};

[Serializable]
public ref class CNTKRuntimeException : CNTKException
{
public:
    CNTKRuntimeException() : CNTKException()
    {}

    CNTKRuntimeException(String^ message, String^ callstack) : CNTKException(message, callstack)
    {}

protected:

    CNTKRuntimeException(SerializationInfo^ info, StreamingContext context) : CNTKException(info, context)
    {}
};

[Serializable]
public ref class CNTKLogicErrorException : CNTKException
{
public:
    CNTKLogicErrorException() : CNTKException()
    {}

    CNTKLogicErrorException(String^ message, String^ callstack) : CNTKException(message, callstack)
    {}

protected:

    CNTKLogicErrorException(SerializationInfo^ info, StreamingContext context) : CNTKException(info, context)
    {}
};

[Serializable]
public ref class CNTKInvalidArgumentException : CNTKException
{
public:
    CNTKInvalidArgumentException() : CNTKException()
    {}

    CNTKInvalidArgumentException(String^ message, String^ callstack) : CNTKException(message, callstack)
    {}

protected:

    CNTKInvalidArgumentException(SerializationInfo^ info, StreamingContext context) : CNTKException(info, context)
    {}
};

[Serializable]
public ref class CNTKBadAllocException : CNTKException
{
public:
    CNTKBadAllocException() : CNTKException()
    {}

    CNTKBadAllocException(String^ message) : CNTKException(message)
    {}

protected:

    CNTKBadAllocException(SerializationInfo^ info, StreamingContext context) : CNTKException(info, context)
    {}
};


}}}}}
