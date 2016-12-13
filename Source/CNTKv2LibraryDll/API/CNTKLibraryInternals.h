//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Contains internals used for defining the CNTKLibrary.h APIs
//

#pragma once

#ifdef SWIG
#define final
#define explicit
#define static_assert(condition, message)
#define __attribute__(x)
#endif

#ifdef _WIN32
#ifdef CNTKV2LIBRARYDLL
#define CNTK_API __declspec(dllexport)
#else
#define CNTK_API __declspec(dllimport)
#endif
#define _SCL_SECURE_NO_WARNINGS
#else // no DLLs on Linux
#define CNTK_API
#endif

#include <memory>
#include <vector>
#include <array>
#include <stdarg.h>
#include <assert.h>
#include <atomic>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <stdlib.h>
#include <string.h>

#pragma warning(disable: 4702 4127)

// Forward declarations
namespace Microsoft { namespace MSR { namespace CNTK {
    template <typename ElemType>
    class Matrix;

    template <typename ElemType>
    class TensorView;

    class ComputationNetwork;

    template <typename ElemType>
    class ComputationNetworkBuilder;

    template <typename ElementType>
    class ComputationNode;

    class ComputationNodeBase;
    typedef std::shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;
}}}

// TODO: The following should be reconciled with the equivalent code in the CNTK implementation

#ifndef _MSC_VER
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
static inline wchar_t* _wcsdup(const wchar_t *s)
{
    return ::wcsdup(s);
}
#endif

namespace CNTK
{

#define UNUSED(x) (void)(x) // for variables that are, e.g., only used in _DEBUG builds

#ifdef _MSC_VER
#define __declspec_noreturn __declspec(noreturn)
#else
#define __declspec_noreturn __attribute__((noreturn))
#endif

#pragma warning(push)
#pragma warning(disable : 4996)
#ifndef _MSC_VER // TODO: what is the correct trigger for gcc?
    template <class E>
    __declspec_noreturn void ThrowFormatted(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif

    template <class E>
    __declspec_noreturn inline void ThrowFormatted(const char* format, ...)
    {
        va_list args;
        va_start(args, format);

        char buffer[1024] = { 0 }; // Note: pre-VS2015 vsnprintf() is not standards-compliant and may not add a terminator
        int written = vsnprintf(buffer, _countof(buffer) - 1, format, args); // -1 because pre-VS2015 vsnprintf() does not always write a 0-terminator
        // TODO: In case of EILSEQ error, choose between just outputting the raw format itself vs. continuing the half-completed buffer
        //if (written < 0) // an invalid wide-string conversion may lead to EILSEQ
        //    strncpy(buffer, format, _countof(buffer)
        UNUSED(written); // pre-VS2015 vsnprintf() returns -1 in case of overflow, instead of the #characters written
        if (strlen(buffer)/*written*/ >= (int)_countof(buffer) - 2)
            sprintf(buffer + _countof(buffer) - 4, "...");

        // TODO: Should use ExceptionWithCallStack; temporarily using std::exception to avoid duplicating headers
        //throw ExceptionWithCallStack<E>(buffer, ExceptionWithCallStack<E>::GetCallStack(/*skipLevels=*/2, /*makeFunctionNamesStandOut=*/true));
        throw E(buffer);
    }
#pragma warning(pop)

    // RuntimeError - throw a std::runtime_error with a formatted error string
#ifndef _MSC_VER // gcc __attribute__((format(printf())) does not percolate through variadic templates; so must go the macro route
#ifndef RuntimeError
#define RuntimeError ThrowFormatted<std::runtime_error>
#endif
#ifndef LogicError
#define LogicError ThrowFormatted<std::logic_error>
#endif
#ifndef InvalidArgument
#define InvalidArgument ThrowFormatted<std::invalid_argument>
#endif
#else
    template <class... _Types>
    __declspec_noreturn inline void RuntimeError(const char* format, _Types&&... _Args)
    {
        ThrowFormatted<std::runtime_error>(format, std::forward<_Types>(_Args)...);
    }
    template <class... _Types>
    __declspec_noreturn inline void LogicError(const char* format, _Types&&... _Args)
    {
        ThrowFormatted<std::logic_error>(format, std::forward<_Types>(_Args)...);
    }
    template <class... _Types>
    __declspec_noreturn inline void InvalidArgument(const char* format, _Types&&... _Args)
    {
        ThrowFormatted<std::invalid_argument>(format, std::forward<_Types>(_Args)...);
    }
#endif

#ifndef NOT_IMPLEMENTED
#define NOT_IMPLEMENTED                                                                                                              \
    {                                                                                                                                \
        fprintf(stderr, "Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__); \
        CNTK::LogicError("Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__);      \
    }
#endif
}

namespace CNTK
{
    // Forward declarations
    class CompositeFunction;
    class Function;
    class Variable;
    class Axis;

    // Similar to make_shared except that it associates a custom deleter with the shared_ptr to ensure
    // that objects are deleted on the same side of the library DLL where they are allocated
    template <typename T, typename ...CtorArgTypes>
    inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs)
    {
        auto objPtr = new T(std::forward<CtorArgTypes>(ctorArgs)...);
        return std::shared_ptr<T>(objPtr, [](T* ptr) { delete ptr; });
    }

    // Forward declarations
    class NDArrayView;
    typedef std::shared_ptr<NDArrayView> NDArrayViewPtr;

    class NDMask;
    typedef std::shared_ptr<NDMask> NDMaskPtr;

    class Value;
    typedef std::shared_ptr<Value> ValuePtr;

    class Function;
    typedef std::shared_ptr<Function> FunctionPtr;

    class Learner;
    typedef std::shared_ptr<Learner> LearnerPtr;

    class Dictionary;

    class MinibatchSource;
    typedef std::shared_ptr<MinibatchSource> MinibatchSourcePtr;

    namespace Internal
    {
        // Create a new Function instance which just passes through specified list of 'operands'.
        CNTK_API FunctionPtr Combine(const std::vector<Variable>& operands, const std::wstring& name = L"");

        CNTK_API FunctionPtr IsWithin(const Variable& operand, int offset, const std::wstring& name = L"");
        CNTK_API FunctionPtr PackedIndex(const Variable& operand, const Variable& index, const std::wstring& name = L"");
        CNTK_API FunctionPtr GatherPacked(const Variable& operand, const Variable& packedIndex, const std::wstring& name = L"");
        CNTK_API FunctionPtr ScatterPacked(const Variable& operand, const Variable& packedIndex, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr ZeroesWithDynamicAxesLike(const Variable& operand);
        CNTK_API FunctionPtr Where(const Variable& condition, const std::vector<Axis>& newDynamicAxes, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::vector<Axis>& newDynamicAxes, const std::wstring& name = L"");
        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::vector<Axis>& newDynamicAxes, const std::wstring& name = L"");
        CNTK_API FunctionPtr Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, const std::wstring& name = L"");

        CNTK_API size_t NewUniqueId();

        CNTK_API void EnableReversingTensorShapesInErrorMessages();
        bool IsReversingTensorShapesInErrorMessagesEnabled();

        CNTK_API void AlwaysAllowSettingDefaultDevice();
        bool IsSettingDefaultDeviceAlwaysAllowed();
    }
}
