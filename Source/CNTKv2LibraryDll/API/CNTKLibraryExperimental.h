//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This is the main header of the CNTK library API containing the entire public API definition. 
//

#pragma once

#ifdef SWIG
#define final
#define explicit
#define static_assert(condition, message)
#endif

#include "CNTKLibrary.h"

///
/// Experimental features in CNTK library. 
/// Please be aware that these are subject to frequent changes and even removal.
///
namespace CNTK { namespace Experimental {

    class DeserializerFactory
    {
    public:
        virtual DataDeserializerPtr operator()(const std::wstring&) const = 0;
        virtual ~DeserializerFactory() = default;
    };

    typedef std::shared_ptr<DeserializerFactory> DeserializerFactoryPtr;

    CNTK_API void RegisterDeserializerFactory(DeserializerFactoryPtr callbackPtr);
    CNTK_API DeserializerFactoryPtr GetDeserializerFactory();
}}
