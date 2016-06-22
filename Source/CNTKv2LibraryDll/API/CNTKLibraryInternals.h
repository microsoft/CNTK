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

    namespace _Internal
    {
        //  A reference counter to be used as the base class for all reference counted types.
        class _ReferenceCounter
        {
        public:

            //  Constructor.
            _ReferenceCounter() : m_rc(0) {}

            //  Destructor.
            virtual ~_ReferenceCounter() {}

            // Add a reference. 
            // Thread-safe.
            size_t AddReference()
            {
                return ++m_rc;
            }

            // Remove a reference. 
            // Thread-safe.
            size_t RemoveReference()
            {
                assert(m_rc.load() > 0);
                return --m_rc;
            }

            // Return the reference count value
            size_t GetReferenceCount()
            {
                return m_rc.load();
            }

        private:
            std::atomic<size_t> m_rc;
        };

        // A smart pointer to a reference counted object
        // T must be a type derived from _Reference_counter
        template <class T>
        class _ReferenceCounterSharedPtr final
        {
            typedef void(*_ReferenceCounterDeleter)(_ReferenceCounter* obj);

        public:

            // Constructor
            _ReferenceCounterSharedPtr(T* ptr = nullptr, _ReferenceCounterDeleter deleter = nullptr) : m_objPtr(ptr), m_deleter(deleter)
            {
                Init();
            }

            // Copy constructor
            _ReferenceCounterSharedPtr(const _ReferenceCounterSharedPtr& other) : m_objPtr(nullptr), m_deleter(nullptr)
            {
                *this = other;
            }

            // Move constructor
            _ReferenceCounterSharedPtr(_ReferenceCounterSharedPtr&& other) : m_objPtr(nullptr), m_deleter(nullptr)
            {
                *this = std::move(other);
            }

            // Destructor
            ~_ReferenceCounterSharedPtr()
            {
                UnInitialize(m_objPtr, m_deleter);
            }

            // Assignment operator
            _ReferenceCounterSharedPtr& operator=(const _ReferenceCounterSharedPtr& other)
            {
                if (this != &other)
                {
                    T* oldPtr = m_objPtr;
                    _ReferenceCounterDeleter oldDeleter = m_deleter;

                    m_objPtr = other.m_objPtr;
                    m_deleter = other.m_deleter;
                    Init();

                    UnInitialize(oldPtr, oldDeleter);
                }

                return *this;
            }

            // Move-assignment operator
            _ReferenceCounterSharedPtr& operator=(_ReferenceCounterSharedPtr&& other)
            {
                assert(this != &other);

                T* oldPtr = m_objPtr;
                _ReferenceCounterDeleter oldDeleter = m_deleter;

                m_objPtr = other.m_objPtr;
                m_deleter = other.m_deleter;
                // No change to ref-count of the adopted pointer.

                other.m_objPtr = nullptr;
                other.m_deleter = nullptr;

                UnInitialize(oldPtr, oldDeleter);

                return *this;
            }

            // Conversion to a ReferenceCountedSharedPtr instance of a base type
            template <typename Base, typename std::enable_if<std::is_base_of<Base, T>::value>::type* = nullptr>
            operator _ReferenceCounterSharedPtr<Base>()
            {
                return _ReferenceCounterSharedPtr<Base>(m_objPtr, m_deleter);
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
            void Init()
            {
                static_assert(std::is_base_of<_ReferenceCounter, T>::value, "_ReferenceCounterSharedPtr<T> can only be used when _ReferenceCounter is a base type of T!");

                if (m_objPtr != nullptr)
                    reinterpret_cast<_ReferenceCounter*>(m_objPtr)->AddReference();
            }

            static void UnInitialize(T* objPtr, _ReferenceCounterDeleter deleter)
            {
                static_assert(std::is_base_of<_ReferenceCounter, T>::value, "_ReferenceCounterSharedPtr<T> can only be used when _ReferenceCounter is a base type of T!");

                if (objPtr != nullptr)
                {
                    size_t refCountRemaining = reinterpret_cast<_ReferenceCounter*>(objPtr)->RemoveReference();
                    if (refCountRemaining == 0)
                    {
                        if (deleter != nullptr)
                            deleter(reinterpret_cast<_ReferenceCounter*>(objPtr));
                        else
                            delete objPtr;
                    }
                }
            }

        private:
            T* m_objPtr;
            _ReferenceCounterDeleter m_deleter;
        };

        template <typename T>
        bool operator==(const _ReferenceCounterSharedPtr<T>& first, const _ReferenceCounterSharedPtr<T>& second)
        {
            return first.GetPtr() == second.GetPtr();
        }

        // A simple vector implementation with a C ABI to allow usage across the library DLL boundary
        // as STL vectors cannot be used across the DLL boundary
        template <typename T>
        class CNTK_API _SimpleVector final
        {
            template <typename ValueType>
            friend CNTK_API bool operator==(const _SimpleVector<ValueType>& first, const _SimpleVector<ValueType>& second);

            friend class CNTK::Function;

        public:
            _SimpleVector();
            _SimpleVector(size_t numElements, const T& initVal = T());
            ~_SimpleVector();

            _SimpleVector(const _SimpleVector& other);
            _SimpleVector& operator=(const _SimpleVector& other);

            _SimpleVector(_SimpleVector&& other);
            _SimpleVector& operator=(_SimpleVector&& other);

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
                        RuntimeError("A _SimpleVector with duplicate elements cannot be converted to an unordered_set");
                }

                return retSet;
            }

            template <typename ContainerType, typename std::enable_if<std::is_same<ContainerType, std::vector<T>>::value ||
                                                                      std::is_same<ContainerType, std::initializer_list<T>>::value ||
                                                                      std::is_same<ContainerType, std::array<T, sizeof(ContainerType) / sizeof(T)>>::value>::type* = nullptr>
            static _SimpleVector<T> CreateSimpleVector(const ContainerType& initList)
            {
                _SimpleVector<T> simpleVector(initList.size());
                std::copy(initList.begin(), initList.end(), simpleVector.Data());

                return simpleVector;
            }

        private:
            std::vector<T>* m_vector;
        };

        template <typename ValueType>
        CNTK_API bool operator==(const _SimpleVector<ValueType>& first, const _SimpleVector<ValueType>& second);

        template <typename ValueType>
        bool operator!=(const _SimpleVector<ValueType>& first, const _SimpleVector<ValueType>& second)
        {
            return !(first == second);
        }

        // A simple set implementation with a C ABI to allow usage across the library DLL boundary
        // as STL sets cannot be used across the DLL boundary
        template <typename KeyType>
        class CNTK_API _SimpleSet final
        {
            friend class CNTK::CompositeFunction;

            template <typename T>
            friend CNTK_API bool operator==(const _SimpleSet<T>& first, const _SimpleSet<T>& second);

        public:
            _SimpleSet();
            ~_SimpleSet();

            _SimpleSet(const _SimpleSet& other);
            _SimpleSet& operator=(const _SimpleSet& other);

            _SimpleSet(_SimpleSet&& other);
            _SimpleSet& operator=(_SimpleSet&& other);

            bool Insert(const KeyType& key);
            bool Contains(const KeyType& key) const;

            size_t Size() const;

            operator _SimpleVector<KeyType>() const;

            operator std::unordered_set<KeyType>() const
            {
                return ((_SimpleVector<KeyType>)(*this)).GetAsUnorderedSet();
            }

            static _SimpleSet<KeyType> CreateSimpleSet(const std::unordered_set<KeyType>& initSet)
            {
                _SimpleSet<KeyType> simpleSet;
                for (auto iter = initSet.begin(); iter != initSet.end(); ++iter)
                    simpleSet.Insert(*iter);

                return simpleSet;
            }

        private:
            std::unordered_set<KeyType>* m_set;
        };

        template <typename KeyType>
        CNTK_API bool operator==(const _SimpleSet<KeyType>& first, const _SimpleSet<KeyType>& second);

        template <typename KeyType>
        bool operator!=(const _SimpleSet<KeyType>& first, const _SimpleSet<KeyType>& second)
        {
            return !(first == second);
        }

        // A simple map implementation with a C ABI to allow usage across the library DLL boundary
        // as STL maps cannot be used across the DLL boundary
        template <typename KeyType, typename ValueType>
        class CNTK_API _SimpleMap final
        {
            friend class CNTK::CompositeFunction;
            friend class CNTK::Function;

        public:
            _SimpleMap();
            ~_SimpleMap();

            _SimpleMap(const _SimpleMap& other);
            _SimpleMap& operator=(const _SimpleMap& other);

            _SimpleMap(_SimpleMap&& other);
            _SimpleMap& operator=(_SimpleMap&& other);

            ValueType& operator[](const KeyType& key);
            const ValueType& operator[](const KeyType& key) const;

            bool Insert(const KeyType& key, const ValueType& value);
            bool Contains(const KeyType& key) const;
            size_t Size() const;

            _SimpleSet<KeyType> Keys() const;

            static _SimpleMap<KeyType, ValueType> CreateSimpleMap(const std::unordered_map<KeyType, ValueType>& initMap)
            {
                _SimpleMap<KeyType, ValueType> simpleMap;
                for (auto iter = initMap.begin(); iter != initMap.end(); ++iter)
                    simpleMap.Insert(iter->first, iter->second);

                return simpleMap;
            }

        private:
            std::unordered_map<KeyType, ValueType>* m_map;
        };
    }

    // Forward declarations
    class NDArrayView;
    typedef _Internal::_ReferenceCounterSharedPtr<NDArrayView> NDArrayViewPtr;

    class NDMask;
    typedef _Internal::_ReferenceCounterSharedPtr<NDMask> NDMaskPtr;

    class Value;
    typedef _Internal::_ReferenceCounterSharedPtr<Value> ValuePtr;

    class Function;
    typedef _Internal::_ReferenceCounterSharedPtr<Function> FunctionPtr;

    inline wchar_t* CopyString(const wchar_t* source)
    {
        size_t len = wcslen(source) + 1;
        wchar_t* copy = new wchar_t[len];
#ifdef _WIN32
        wcscpy_s(copy, len, source);
#else
        wcscpy(copy, source);
#endif
        return copy;
    }
}

namespace std {
    template <typename T>
    struct hash<CNTK::_Internal::_ReferenceCounterSharedPtr<T>>
    {
        size_t operator()(const CNTK::_Internal::_ReferenceCounterSharedPtr<T>& x) const
        {
            return std::hash<const void*>()(x.GetPtr());
        }
    };
}
