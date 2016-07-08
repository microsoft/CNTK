//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Contains internals used for defining the CNTKLibrary.h APIs
//

#pragma once

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
#define RuntimeError ThrowFormatted<std::runtime_error>
#define LogicError ThrowFormatted<std::logic_error>
#define InvalidArgument ThrowFormatted<std::invalid_argument>
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
        LogicError("Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__);      \
    }
#endif
}

namespace CNTK
{
    // Forward declarations
    class CompositeFunction;
    class Function;
    class Variable;

    namespace Internal
    {
        template <typename T>
        class ReferenceCountedPtr;
    }

    template <typename T, typename ...CtorArgTypes>
    static Internal::ReferenceCountedPtr<T> MakeReferenceCountedObject(CtorArgTypes&& ...ctorArgs);

    namespace Internal
    {
        //  A reference count to be used as the base class for all reference counted types.
        class CNTK_API ReferenceCount
        {
            typedef void(*ReferenceCountedObjectDeleter)(ReferenceCount* obj);

        public:

            ReferenceCount();
            virtual ~ReferenceCount();

            size_t AddReference();
            size_t RemoveReference();
            size_t GetReferenceCount();
            void SetDeleter(ReferenceCountedObjectDeleter deleter);
            ReferenceCountedObjectDeleter GetDeleter() const;

        private:
            std::atomic<size_t>* m_rc;
            ReferenceCountedObjectDeleter m_deleter;
        };

        // A smart pointer to a reference counted object
        // T must be a type derived from ReferenceCount
        template <class T>
        class CNTK_API ReferenceCountedPtr final
        {
            friend class Variable;
            friend class Function;

            template <typename T, typename ...CtorArgTypes>
            friend static ReferenceCountedPtr<T> CNTK::MakeReferenceCountedObject(CtorArgTypes&& ...ctorArgs);

            template <typename U>
            friend class ReferenceCountedPtr;

        public:
            
            ReferenceCountedPtr(nullptr_t) : ReferenceCountedPtr(reinterpret_cast<T*>(nullptr)) {}
            ReferenceCountedPtr() : ReferenceCountedPtr(nullptr) {}

            ReferenceCountedPtr(const ReferenceCountedPtr& other) : m_objPtr(nullptr)
            {
                *this = other;
            }

            ReferenceCountedPtr(ReferenceCountedPtr&& other) : m_objPtr(nullptr)
            {
                *this = std::move(other);
            }

            ~ReferenceCountedPtr()
            {
                DeleteReferenceIfNeeded(m_objPtr);
            }

            ReferenceCountedPtr& operator=(const ReferenceCountedPtr& other)
            {
                if (this != &other)
                {
                    T* oldPtr = m_objPtr;
                    m_objPtr = other.m_objPtr;
                    AddReferenceIfNeeded();

                    DeleteReferenceIfNeeded(oldPtr);
                }

                return *this;
            }

            ReferenceCountedPtr& operator=(ReferenceCountedPtr&& other)
            {
                assert(this != &other);

                T* oldPtr = m_objPtr;
                m_objPtr = other.m_objPtr;

                // No change to ref-count of the adopted pointer.

                other.m_objPtr = nullptr;

                DeleteReferenceIfNeeded(oldPtr);

                return *this;
            }

            // Conversion to a ReferenceCountedSharedPtr instance of a base type
            template <typename Base, typename std::enable_if<std::is_base_of<Base, T>::value>::type* = nullptr>
            operator ReferenceCountedPtr<Base>()
            {
                return ReferenceCountedPtr<Base>(m_objPtr);
            }

            T* operator->() const
            {
                return m_objPtr;
            }

            T& operator*() const
            {
                return *m_objPtr;
            }

            operator T*() const
            {
                return m_objPtr;
            }

            T* GetPtr() const
            {
                return m_objPtr;
            }

        private:
            ReferenceCountedPtr(T* ptr) : m_objPtr(ptr)
            {
                static_assert(std::is_base_of<ReferenceCount, T>::value, "ReferenceCountedPtr<T> can only be used when ReferenceCount is a base type of T!");
                AddReferenceIfNeeded();
            }

            void AddReferenceIfNeeded()
            {
                if (m_objPtr != nullptr)
                {
                    assert(m_objPtr->GetDeleter() != nullptr);
                    reinterpret_cast<ReferenceCount*>(m_objPtr)->AddReference();
                }
            }

            static void DeleteReferenceIfNeeded(T* objPtr)
            {
                if (objPtr != nullptr)
                {
                    size_t refCountRemaining = reinterpret_cast<ReferenceCount*>(objPtr)->RemoveReference();
                    if (refCountRemaining == 0)
                    {
                        auto deleter = objPtr->GetDeleter();
                        assert(deleter != nullptr);
                        deleter(reinterpret_cast<ReferenceCount*>(objPtr));
                    }
                }
            }

        private:
            T* m_objPtr;
        };

        template <typename T>
        bool operator==(const ReferenceCountedPtr<T>& first, const ReferenceCountedPtr<T>& second)
        {
            return first.GetPtr() == second.GetPtr();
        }

        // A wrapper around the STL vector implementation with a safe ABI to allow usage across the library DLL boundary
        // as STL vectors cannot be used across the DLL boundary
        template <typename T>
        class CNTK_API SimpleVector final
        {
            template <typename ValueType>
            friend CNTK_API bool operator==(const SimpleVector<ValueType>& first, const SimpleVector<ValueType>& second);

            friend class CNTK::Function;

        public:
            SimpleVector();

            template <typename ContainerType, typename std::enable_if<std::is_same<ContainerType, std::vector<T>>::value ||
                                                                      std::is_same<ContainerType, std::initializer_list<T>>::value ||
                                                                      std::is_same<ContainerType, std::array<T, sizeof(ContainerType) / sizeof(T)>>::value>::type* = nullptr>
            SimpleVector(const ContainerType& initList)
                : SimpleVector(initList.size())
            {
                std::copy(initList.begin(), initList.end(), Data());
            }

            SimpleVector(size_t numElements, const T& initVal = T());
            ~SimpleVector();

            SimpleVector(const SimpleVector& other);
            SimpleVector& operator=(const SimpleVector& other);

            SimpleVector(SimpleVector&& other);
            SimpleVector& operator=(SimpleVector&& other);

            T& operator[](size_t idx);
            const T& operator[](size_t idx) const;

            size_t Size() const;

            T* Data();
            const T* Data() const;

            void PushBack(const T& value);
            void PushBack(T&& value);

            operator std::vector<T>() const
            {
                std::vector<T> retVector(Size());
                for (size_t i = 0; i < Size(); ++i)
                    retVector[i] = this->operator[](i);

                return retVector;
            }

            std::unordered_set<T> GetAsUnorderedSet(bool ensureUnique = true)
            {
                std::unordered_set<T> retSet;
                for (size_t i = 0; i < Size(); ++i)
                {
                    auto insertRet = retSet.insert(this->operator[](i));
                    if (ensureUnique && !insertRet.second)
                        RuntimeError("A SimpleVector with duplicate elements cannot be converted to an unordered_set");
                }

                return retSet;
            }

        private:
            std::vector<T>* m_vector;
        };

        template <typename ValueType>
        CNTK_API bool operator==(const SimpleVector<ValueType>& first, const SimpleVector<ValueType>& second);

        template <typename ValueType>
        bool operator!=(const SimpleVector<ValueType>& first, const SimpleVector<ValueType>& second)
        {
            return !(first == second);
        }

        // A wrapper around the STL set implementation with a safe ABI to allow usage across the library DLL boundary
        // as STL sets cannot be used across the DLL boundary
        template <typename KeyType>
        class CNTK_API SimpleSet final
        {
            friend class CNTK::CompositeFunction;

            template <typename T>
            friend CNTK_API bool operator==(const SimpleSet<T>& first, const SimpleSet<T>& second);

        public:
            SimpleSet();
            ~SimpleSet();

            SimpleSet(const SimpleSet& other);
            SimpleSet& operator=(const SimpleSet& other);

            SimpleSet(SimpleSet&& other);
            SimpleSet& operator=(SimpleSet&& other);

            bool Insert(const KeyType& key);
            bool Contains(const KeyType& key) const;

            size_t Size() const;

            operator SimpleVector<KeyType>() const;

            operator std::unordered_set<KeyType>() const
            {
                return ((SimpleVector<KeyType>)(*this)).GetAsUnorderedSet();
            }

            static SimpleSet<KeyType> CreateSimpleSet(const std::unordered_set<KeyType>& initSet)
            {
                SimpleSet<KeyType> simpleSet;
                for (auto key : initSet)
                    simpleSet.Insert(key);

                return simpleSet;
            }

        private:
            std::unordered_set<KeyType>* m_set;
        };

        template <typename KeyType>
        CNTK_API bool operator==(const SimpleSet<KeyType>& first, const SimpleSet<KeyType>& second);

        template <typename KeyType>
        bool operator!=(const SimpleSet<KeyType>& first, const SimpleSet<KeyType>& second)
        {
            return !(first == second);
        }

        // A wrapper aroound the STL map implementation with a safe ABI to allow usage across the library DLL boundary
        // as STL maps cannot be used across the DLL boundary
        template <typename KeyType, typename ValueType>
        class CNTK_API SimpleMap final
        {
            friend class CNTK::CompositeFunction;
            friend class CNTK::Function;

        public:
            SimpleMap();
            ~SimpleMap();

            SimpleMap(const SimpleMap& other);
            SimpleMap& operator=(const SimpleMap& other);

            SimpleMap(SimpleMap&& other);
            SimpleMap& operator=(SimpleMap&& other);

            ValueType& operator[](const KeyType& key);
            const ValueType& operator[](const KeyType& key) const;

            bool Insert(const KeyType& key, const ValueType& value);
            bool Contains(const KeyType& key) const;
            size_t Size() const;

            SimpleSet<KeyType> Keys() const;

            static SimpleMap<KeyType, ValueType> CreateSimpleMap(const std::unordered_map<KeyType, ValueType>& initMap)
            {
                SimpleMap<KeyType, ValueType> simpleMap;
                for (auto keyValuePair : initMap)
                    simpleMap.Insert(keyValuePair.first, keyValuePair.second);

                return simpleMap;
            }

        private:
            std::unordered_map<KeyType, ValueType>* m_map;
        };
    }

    template <typename T, typename ...CtorArgTypes>
    static Internal::ReferenceCountedPtr<T> MakeReferenceCountedObject(CtorArgTypes&& ...ctorArgs)
    {
        static_assert(std::is_base_of<Internal::ReferenceCount, T>::value, "MakeReferenceCountedObject<T> can only be used when ReferenceCount is a base type of T!");

        auto objPtr = new T(std::forward<CtorArgTypes>(ctorArgs)...);
        objPtr->SetDeleter([](Internal::ReferenceCount* ptr) { delete ptr; });

        return Internal::ReferenceCountedPtr<T>(objPtr);
    }

    // Forward declarations
    class NDArrayView;
    typedef Internal::ReferenceCountedPtr<NDArrayView> NDArrayViewPtr;

    class NDMask;
    typedef Internal::ReferenceCountedPtr<NDMask> NDMaskPtr;

    class Value;
    typedef Internal::ReferenceCountedPtr<Value> ValuePtr;

    class Function;
    typedef Internal::ReferenceCountedPtr<Function> FunctionPtr;

    namespace Internal
    {
        CNTK_API FunctionPtr Combine(const Internal::SimpleVector<FunctionPtr>& operands, const std::wstring& name = L"");
    }
}

namespace std {
    template <typename T>
    struct hash<CNTK::Internal::ReferenceCountedPtr<T>>
    {
        size_t operator()(const CNTK::Internal::ReferenceCountedPtr<T>& x) const
        {
            return std::hash<const void*>()(x.GetPtr());
        }
    };
}
