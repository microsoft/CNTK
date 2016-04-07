// BrainScriptObjects.h -- objects that the config parser operates on

#pragma once

#include "Basics.h"
#include "Platform.h" // for noexcept workaround on VS2013

#include <memory>     // for shared_ptr<>
#include <functional> // for function<>
#include <map>
#include <set>

namespace Microsoft { namespace MSR { namespace ScriptableObjects {

using namespace std;
using namespace msra::strfun;         // for wstrprintf()
using namespace Microsoft::MSR::CNTK; // for stuff from Basics.h

// -----------------------------------------------------------------------
// ScriptingException -- base class for any errors thrown by scripting
// It's a runtime_error with an additional virtual function PrintError().
// -----------------------------------------------------------------------

class ScriptingException : public runtime_error
{
public:
    template <typename M>
    ScriptingException(const M &msg)
        : runtime_error(msg)
    {
    }
    virtual void PrintError() const = 0;
};

// -----------------------------------------------------------------------
// Object -- common base class for objects that can be used in config files
// -----------------------------------------------------------------------

// All values that can be used in config files
//  - are heap objects
//     - primitives are wrapped
//     - object pointers are ref-counted shared_ptr, wrapped in ConfigValuePtr (see BrainScriptEvaluator.h)
//  - derive from Object (outside classes get wrapped)
//
// This code supports three kinds of value types:
//  - self-defined classes -> derive from Object, e.g. Expression
//  - classes defined outside -> wrap in a BoxOf object, e.g. String = BoxOf<std::wstring>
//  - C++ primitives like 'double' -> wrap in a Wrapper first then in a BoxOf, e.g. Number = BoxOf<Wrapped<double>>

struct Object
{
    virtual ~Object()
    {
    }
};

// indicates that the object has a name should be set from the expression path

struct HasName
{
    virtual void SetName(const std::wstring &name) = 0;
};

// -----------------------------------------------------------------------
// Wrapped<T> -- wraps non-class primitive C++ type into a class, like 'double'.
// (It can also be used for class types, but better use BoxOf<> below directly.)
// -----------------------------------------------------------------------

template <typename T>
class Wrapped
{
    T value; // meant to be a primitive type
public:
    operator const T &() const
    {
        return value;
    }
    operator T &()
    {
        return value;
    }
    Wrapped(T value)
        : value(value)
    {
    }
    T &operator=(const T &newValue)
    {
        value = newValue;
        return *this;
    }
};
typedef Wrapped<double> Double;
typedef Wrapped<bool> Bool;

// -----------------------------------------------------------------------
// BoxOf<T> -- wraps a pre-defined type, e.g. std::wstring, to derive from Object.
// BoxOf<T> can dynamic_cast to T (e.g. BoxOf<std::wstring> is a std::wstring).
// -----------------------------------------------------------------------

template <class C>
class BoxOf : public Object, public C
{
public:
#if 1
    template <class... _Types>
    BoxOf(_Types &&... _Args)
        : C(forward<_Types>(_Args)...)
    {
    }
#else
    // TODO: change this to variadic templates, then we can instantiate everything we need through this
    BoxOf(const C &val)
        : C(val)
    {
    }
    BoxOf()
    {
    }
#endif
};

// -----------------------------------------------------------------------
// String -- a std::string in config files
// Can cast to std::wstring (done in a way that ConfigValuePtr can also cast to std::wstring).
// -----------------------------------------------------------------------

typedef BoxOf<std::wstring> String;

// -----------------------------------------------------------------------
// ComputationNodeObject -- the 'magic' class that our parser understands for infix operations
// TODO: unify with ComputationNodeBase
// -----------------------------------------------------------------------

class ComputationNodeObject : public Object
{
}; // a base class for all nodes (that has no template parameter)

// -----------------------------------------------------------------------
// HasToString -- trait to indicate an object can print their content
// Derive from HasToString() and implement ToString() method.
// FormatConfigValue() will then return ToString().
// -----------------------------------------------------------------------

struct HasToString
{
    virtual std::wstring ToString() const = 0;

    // some std::string helpers useful for ToString() operations of nested structures
    // TODO: move these out from this header into some more general place (I had to move them here because otherwise CNTKEval failed to compile)
    static std::wstring IndentString(std::wstring s, size_t indent)
    {
        const std::wstring prefix(indent, L' ');
        size_t pos = 0;
        for (;;)
        {
            s.insert(pos, prefix);
            pos = s.find(L'\n', pos + 2);
            if (pos == std::wstring::npos)
                return s;
            pos++;
        }
    }
    static std::wstring NestString(std::wstring s, wchar_t open, bool newline, wchar_t close)
    {
        std::wstring result = IndentString(s, 2);
        if (newline) // have a new line after the open symbol
            result = L" \n" + result + L"\n ";
        else
            result.append(L"  ");
        result.front() = open;
        result.back() = close;
        return result;
    }
};

// -----------------------------------------------------------------------
// WithTags -- trait to give an object a set of tag strings
// -----------------------------------------------------------------------

class WithTags : std::set<std::wstring>
{
public:
    WithTags()
    {
    }
    bool SetTag(const std::wstring &tag)
    {
        auto res = insert(tag);
        return res.second; // true if was not there before
    }
    bool ClearTag(const std::wstring &tag)
    {
        auto iter = find(tag);
        if (iter == end())
            return false;
        erase(iter);
        return true; // indicates that we used to have this tag
    }
    bool HasTag(const std::wstring &tag) const
    {
        return find(tag) != end();
    }
    const std::set<std::wstring>& GetTags() const
    {
        return *this;
    }
};

// =======================================================================
// ConfigValuePtr -- shared pointer to a config value
// =======================================================================

// A ConfigValuePtr holds the value of a configuration variable.
//  - specifically, it holds a shared_ptr to a strongly typed C++ object
//  - ConfigValuePtrs are immutable when consumed.
//
// All configuration values, that is, values that can be held by a ConfigValuePtr, derive from BS::Object.
// To get a shared_ptr<T> of an expected type T, type-cast the ConfigValuePtr to it.
// To get the value of a copyable type like T=double or std::wstring, type-cast to T directly.
//
// ConfigValuePtrs are evaluated on-demand upon first retrieval:
//  - initially, a ConfigValuePtr would hold a Thunk; that is, a lambda that computes (resolves) the value
//  - upon first use, the Thunk is invoked to compute the value, which will then *replace* the Thunk
//  - any consumer of a ConfigValuePtr will only ever see the resolved value, since any access for consumption will force it to be resolved
//  - a resolved ConfigValuePtr is immutable
//
// On-demand evaluation is critical to the semantics of this entire configuration system.
// A configuration is but one big expression (of nested records), but some evaluations cause side effects (such as saving a model), and some expressions may not even be in use at all.
// Thus, we must use on-demand evaluation in order to ensure that side effects are only executed when desired.
//
// Further, to ensure a Thunk is executed at most once (otherwise we may get the same side-effect multiple times),
// an unresolved ConfigValuePtr can only live in a single place. This means,
//  - an unresolved ConfigValuePtr (i.e. one holding a Thunk) cannot be copied (while resolved ones are immutable and can be copied freely)
//  - it can be moved (std::move()) during creation
//  - after creation, it should only live in a known location from which it can be retrieved; specifically:
//     - ConfigRecord entries
//     - ConfigArrays elements
//     - ConfigLambdas (default values of named arguments)

struct IConfigRecord;
class ConfigArray;

// TODO: separate this out from BrainScript to an interface that still does type casts--possible?
class ConfigValuePtr : public shared_ptr<Object>
{
    function<void(const std::wstring &)> failfn; // function to call in case of failure due to this value
    std::wstring expressionName;                 // the expression name reflects the path to reach this expression in the (possibly dynamically macro-expanded) expression tree. Used for naming ComputationNodes.

    // Thunk for resolving a value. This Object represents a function that returns a ConfigValuePtr; call to resolve a deferred value
    class Thunk : public Object
    {
        function<ConfigValuePtr()> f;                // the function to compute the value
        bool currentlyResolving;                     // set during resolution phase, to detect circular references
        function<void(const std::wstring &)> failfn; // function to call in case of failure due to this value
    public:
        Thunk(function<ConfigValuePtr()> f, const function<void(const std::wstring &)> &failfn)
            : f(f), failfn(failfn), currentlyResolving(false)
        {
        }
        ConfigValuePtr ResolveValue()
        {
            if (currentlyResolving) // detect circular references (infinite recursion)
                failfn(L"circular reference (expression to compute identifier's value uses the identifier's value)");
            currentlyResolving = true; // can't run from inside ourselves
            return f();
            // no need to reset currentlyResolving because this object gets replaced and thus deleted anyway
        }
    };
    Thunk *GetThunk() const
    {
        return dynamic_cast<Thunk *>(get());
    } // get Thunk object or nullptr if already resolved
public:
    // --- assignment and copy/move constructors

    ConfigValuePtr()
    {
    } // (formally needed somehow)
    ConfigValuePtr(const shared_ptr<Object> &p, const function<void(const std::wstring &)> &failfn, const std::wstring &expressionName)
        : shared_ptr<Object>(p), failfn(failfn), expressionName(expressionName)
    {
    }
    // ConfigValuePtr(const function<ConfigValuePtr()> & f, TextLocation location, const std::wstring & expressionName) : shared_ptr<Object>(make_shared<Thunk>(f, location)), location(location), expressionName(expressionName) { }
    static ConfigValuePtr MakeThunk(const function<ConfigValuePtr()> &f, const function<void(const std::wstring &)> &failfn, const std::wstring &expressionName)
    {
        return ConfigValuePtr(make_shared<Thunk>(f, failfn), failfn, expressionName);
    }
    // TODO: somehow the constructor overload from Thunk function fails to compile, so for now use MakeThunk instead

    ConfigValuePtr(const ConfigValuePtr &other)
    {
        *this = other;
    }
    ConfigValuePtr(ConfigValuePtr &&other) noexcept
    {
        *this = move(other);
    }
    void operator=(const ConfigValuePtr &other)
    {
        if (other.GetThunk()) // unresolved ConfigValuePtrs are not copyable, only movable
            Microsoft::MSR::CNTK::LogicError("ConfigValuePtr::operator=() on unresolved object '%ls'; ConfigValuePtr is not assignable until resolved", expressionName.empty() ? L"(unassigned)" : expressionName.c_str());
        (shared_ptr<Object> &) *this = other;
        failfn = other.failfn;
        expressionName = other.expressionName;
    }
    void operator=(ConfigValuePtr &&other) noexcept
    {
        failfn = move(other.failfn);
        expressionName = move(other.expressionName);
        (shared_ptr<Object> &) *this = move(other);
    }

    void Fail(const std::wstring &msg) const
    {
        failfn(msg);
    }
    const function<void(const std::wstring &)> &GetFailFn() const
    {
        return failfn;
    } // if you need to pass on the fail function

    // --- retrieving values by type cast

    // access as a reference, that is, as a shared_ptr<T>   --use this for Objects
    template <typename T>
    operator shared_ptr<T>() const
    {
        return AsPtr<T>();
    }
    // access as a (const & to) value  --use this for primitive types (also works to get a const std::wstring & from a String)
    template <typename T>
    operator T() const
    {
        return AsRef<T>();
    }
    // Linux gcc barfs on this ^^ for 'us = (double)((std::wstring)arg).size();' due to some ambiguity error (while it works fine with Visual Studio).
    // If you encounter this, instead say 'us = (double)((const std::wstring&)arg).size();' with a &. Don't forget the const (I have seen broken typecasts without).
    operator const IConfigRecord &() const
    {
        return AsRef<IConfigRecord>();
    }
    operator const ConfigArray &() const
    {
        return AsRef<ConfigArray>();
    }
    operator const std::wstring &() const
    {
        return AsRef<std::wstring>();
    } // somehow operator T() does not work here, still giving ambiguous messages. This makes it work. Probably not generic. Need to fix this.
    operator double() const
    {
        return AsRef<Double>();
    }
    operator float() const
    {
        return (float) AsRef<Double>();
    }
    operator bool() const
    {
        return AsRef<Bool>();
    }
    template <typename INT>
    INT AsInt() const
    {
        double val = AsRef<Double>();
        INT ival = (INT) val;
        const wchar_t *type = L"size_t";
        // TODO: there is some duplication of type checking; can we unify that?
        if (ival != val)
            Fail(wstrprintf(L"expected expression of type %ls instead of floating-point value %f", type, val));
        return ival;
    }
    operator size_t() const
    {
        return AsInt<size_t>();
    }
    operator int() const
    {
        return AsInt<int>();
    }

    // --- access functions

    template <class C>
    bool Is() const
    {
        EnsureIsResolved();
        const auto p = dynamic_cast<C *>(get());
        return p != nullptr;
    }
    template <class C>
    const C &AsRef() const // returns reference to the 'value' member. Configs are considered immutable, so return a const&
    {
        // TODO: factor these lines into a separate function
        // Note: since this returns a reference into 'this', you must keep the object you call this on around as long as you use the returned reference
        EnsureIsResolved();
        // const C * wanted = (C *) nullptr; const auto * got = get(); wanted; got;   // allows to see C in the debugger
        const auto p = dynamic_cast<C *>(get());
        if (p == nullptr) // TODO: can we make this look the same as TypeExpected in BrainScriptEvaluator.cpp? We'd need the type name
            Fail(L"config member has wrong type (" + msra::strfun::utf16(typeid(*get()).name()) + L"), expected a " + TypeId<C>());
        return *p;
    }
    template <class C>
    shared_ptr<C> AsPtr() const // returns a shared_ptr cast to the 'value' member
    {
        EnsureIsResolved();
        const auto p = dynamic_pointer_cast<C>(*this);
        if (!p) // TODO: can we make this look the same as TypeExpected in BrainScriptEvaluator.cpp? We'd need the type name
            Fail(L"config member has wrong type (" + msra::strfun::utf16(typeid(*get()).name()) + L"), expected a " + TypeId<C>());
        return p;
    }

    // --- properties

    const char *TypeName() const
    {
        return typeid(*get()).name();
    }
    const std::wstring &GetExpressionName() const
    {
        return expressionName;
    }
    // TODO: ^^ it seems by saving the name in the ConfigValuePtr itself, we don't gain anything; maybe remove again in the future

    // --- methods for resolving the value

    const ConfigValuePtr &ResolveValue() const // (this is const but mutates the value if it resolves)
    {
        // call this when a a member might be as-of-yet unresolved, to evaluate it on-demand
        // get() is a pointer to a Thunk in that case, that is, a function object that yields the value
        const auto thunkp = GetThunk(); // is it a Thunk?
        if (thunkp)                     // value is a Thunk: we need to resolve
        {
            const auto value = thunkp->ResolveValue(); // completely replace ourselves with the actual result. This also releases the Thunk object
            const_cast<ConfigValuePtr &>(*this) = value;
            ResolveValue(); // allow it to return another Thunk...
        }
        return *this; // return ourselves so we can access a value as p_resolved = p->ResolveValue()
    }
    void EnsureIsResolved() const
    {
        if (GetThunk())
            Microsoft::MSR::CNTK::LogicError("ConfigValuePtr: unexpected access to unresolved object; ConfigValuePtrs can only be accessed after resolution");
    }
}; // ConfigValuePtr

// use this for primitive values, double and bool
template <typename T>
static inline ConfigValuePtr MakePrimitiveConfigValuePtr(const T &val, const function<void(const std::wstring &)> &failfn, const std::wstring &exprPath)
{
    return ConfigValuePtr(make_shared<BoxOf<Wrapped<T>>>(val), failfn, exprPath);
}

// -----------------------------------------------------------------------
// IConfigRecord -- config record
// Inside BrainScript, this would be a BS::ConfigRecord, but outside of the
// evaluator, we will only pass it through this interface, to allow for
// extensibility (e.g. Python interfacing).
// Also, Objects themselves can expose this interface to make something accessible.
// -----------------------------------------------------------------------

struct IConfigRecord // any class that exposes config can derive from this
{
    virtual const ConfigValuePtr &operator[](const std::wstring &id) const = 0; // e.g. confRec[L"message"]
    virtual const ConfigValuePtr *Find(const std::wstring &id) const = 0;       // returns nullptr if not found
    virtual std::vector<std::wstring> GetMemberIds() const = 0;                 // returns the names of all members in this record (but not including parent scopes)
    // prettier access if config record is a pointer
    const ConfigValuePtr &Get(const std::wstring &id) const
    {
        return operator[](id);
    } // e.g. confRecPtr->Get(L"message")

    // access with default values
    template <class ValueType>
    ValueType operator()(const std::wstring &id, const ValueType &defaultValue) const // e.g. confRec("message", "hello)
    {
        const auto *valp = Find(id);
        return valp ? *valp : defaultValue;
    }
    inline const IConfigRecord &operator()(const std::wstring &id, const IConfigRecord &defaultValue) const // retrieve a nested ConfigRecord
    {
        const auto *valp = Find(id);
        return valp ? valp->AsRef<IConfigRecord>() : defaultValue;
    }
    std::string operator()(const std::wstring &id, const char *defaultValue) const
    {
        return msra::strfun::utf8(operator()(id, (std::wstring) msra::strfun::utf16(defaultValue)));
    } // special case for narrow strings
    std::wstring operator()(const std::wstring &id, const wchar_t *defaultValue) const
    {
        return operator()(id, std::wstring(defaultValue));
    }

    // -----------------------------------------------------------------------
    // emulation of old CNTK config/NL
    // This allows code written for CNTK config to simply turn ConfigParameters into a template parameter and accept an IConfigRecord.
    // TODO: change all id args to wide strings, then update the code.
    // -----------------------------------------------------------------------

    const ConfigValuePtr &operator()(const std::wstring &id) const;
    template <class T>
    std::vector<T> operator()(const std::wstring &id, const std::vector<T> &defaultValue) const;
    bool ExistsCurrent(const std::wstring &id) const;
    bool Exists(const std::wstring &id) const
    {
        return Find(id) != nullptr;
    }
    bool Match(const std::wstring &id, const std::wstring &compareValue) const;
    bool CanBeConfigRecord(const std::wstring &id) const
    {
        return operator[](id).Is<IConfigRecord>();
    }
    bool CanBeString(const std::wstring &id) const
    {
        return operator[](id).Is<std::wstring>();
    }
    const std::string ConfigName() const;
    static const IConfigRecord &Record();
    template <class V>
    static const std::vector<typename V::value_type> &Array(const V &vec);
};
typedef shared_ptr<struct IConfigRecord> IConfigRecordPtr;

// -----------------------------------------------------------------------
// ConfigRecord -- collection of named config values
// -----------------------------------------------------------------------

class ConfigRecord : public Object, public IConfigRecord // all configuration arguments to class construction, resolved into ConfigValuePtrs
{
    function<void(const std::wstring &)> failfn; // function to call in case of failure due to this value
    // change to ContextInsensitiveMap<ConfigValuePtr>
    std::map<std::wstring, ConfigValuePtr> members;
    IConfigRecordPtr parentScope; // we look up the chain
    ConfigRecord()
    {
    } // forbidden (private) to instantiate without a scope
public:
    // --- creation phase

    ConfigRecord(IConfigRecordPtr parentScope, const function<void(const std::wstring &)> &failfn)
        : parentScope(parentScope), failfn(failfn)
    {
    }
    void Add(const std::wstring &id, const function<void(const std::wstring &)> & /*failfn*/, const ConfigValuePtr &value)
    {
        members[id] = value;
    }
    void Add(const std::wstring &id, const function<void(const std::wstring &)> & /*failfn*/, ConfigValuePtr &&value)
    {
        members[id] = move(value);
    } // use this for unresolved ConfigPtrs
    // TODO: Add() does not yet correctly handle the failfn. It is meant to flag the location of the variable identifier

    // --- usage phase

    // regular lookup: just use record[id] or record(id, L"helpful message what 'id' does")
    // Any unresolved value is resolved at this time, as it is being consumed. Only after resolving a ConfigValuePtr, it can be copied.
    const ConfigValuePtr & /*IConfigRecord::*/ operator[](const std::wstring &id) const // e.g. confRec[L"name"]
    {
        const auto memberIter = members.find(id);
        if (memberIter != members.end())
            return memberIter->second.ResolveValue(); // resolve upon access
        if (!parentScope)                             // not found: if at top scope, we fail
            failfn(L"required parameter '" + id + L"' not found");
        // The failfn will report the location where the dictionary itself was formed.
        // This is because this function is meant to be used by C++ code.
        // When we look up a name by a BrainScript ".FIELD" expression, we will use Find() so we can report the error for the offending FIELD itself.
        return (*parentScope)[id]; // have parent: look it up there
    }
    const ConfigValuePtr * /*IConfigRecord::*/ Find(const std::wstring &id) const // returns nullptr if not found
    {
        auto memberIter = members.find(id);
        if (memberIter == members.end())
            if (parentScope)
                return parentScope->Find(id);
            else
                return nullptr;
        else
            return &memberIter->second.ResolveValue();
    }
    // get member ids; use this when you intend to consume all record entries and do not know the names
    // Note that unlike Find() and operator[], which return parent matches, this only returns entries in this record.
    virtual std::vector<std::wstring> /*IConfigRecord::*/ GetMemberIds() const
    {
        std::vector<std::wstring> ids;
        for (auto &member : members)
            ids.push_back(member.first);
        return ids;
    }
};
typedef shared_ptr<ConfigRecord> ConfigRecordPtr;
// TODO: can ConfigRecordPtr be IConfigRecordPtr?

// create a runtime object from its type --general case
// There can be specializations of this that instantiate objects that do not take ConfigRecords or involve mapping like ComputationNode.
template <typename C>
shared_ptr<Object> MakeRuntimeObject(const IConfigRecordPtr config)
{
    return make_shared<C>(config);
}

// -----------------------------------------------------------------------
// ConfigArray -- an array of config values
// -----------------------------------------------------------------------

// an array is just a std::vector of config values
class ConfigArray : public Object
{
    std::vector<ConfigValuePtr> values;
    int firstIndex;

public:
    ConfigArray()
        : firstIndex(0)
    {
    }
    ConfigArray(int firstIndex, std::vector<ConfigValuePtr> &&values)
        : firstIndex(firstIndex), values(move(values))
    {
    }
    // ConfigArray(ConfigValuePtr && val) : firstIndex(0), values(std::vector<ConfigValuePtr>{ move(val) }) { }
    pair<int, int> GetIndexRange() const
    {
        return make_pair(firstIndex, firstIndex + (int) values.size() - 1);
    }
    // for use as a plain array: get size and verify that index range starts with 0
    template <typename FAILFN>
    size_t GetSize(const FAILFN &Fail) const
    {
        if (firstIndex != 0)
            Fail(L"This array is expected to begin with index 0.");
        return values.size();
    }
    // building the array from expressions: append an element or an array
    void Append(const ConfigValuePtr &value)
    {
        values.push_back(value);
    }
    void Append(ConfigValuePtr &&value)
    {
        values.push_back(move(value));
    } // this appends an unresolved ConfigValuePtr
    void Append(const ConfigArray &other)
    {
        values.insert(values.end(), other.values.begin(), other.values.end());
    }
    // get element at index, including bounds check
    template <typename FAILFN>
    const ConfigValuePtr &At(int index, const FAILFN &Fail /*should report location of the index*/) const
    {
        if (index < firstIndex || index >= firstIndex + values.size())
            Fail(msra::strfun::wstrprintf(L"Index %d out of bounds [%d..%d].", (int) index, (int) firstIndex, (int) (firstIndex + values.size() - 1)));
        return values[(size_t)(index - firstIndex)].ResolveValue(); // resolve upon access
    }
    // get element when knowing that the bounds are correct, e.g. looping over the item range returned by GetItemRange()
    const ConfigValuePtr &At(int index) const
    {
        return At(index, [](const std::wstring &) { LogicError("ConfigArray::At(): Index unexpectedly out of bounds."); });
    }
    // get an entire array into a std::vector. Note that this will force all values to be evaluated.
    template <typename C, typename FAILFN>
    std::vector<C> AsVector(const FAILFN &Fail, bool flatten = false) const
    {
        std::vector<C> res;
        res.reserve(GetSize(Fail));
        for (const auto& valp : values)
        {
            valp.ResolveValue(); // resolve upon access
            if (!flatten || !valp.Is<ConfigArray>())
                res.push_back(valp);
            else // special case: flatten nested vectors (only if 'flatten')
            {
                std::vector<C> subVector = valp.AsRef<ConfigArray>().AsVector<C>(Fail, flatten);
                res.insert(res.end(), subVector.begin(), subVector.end());
            }
        }
        return res;
    }

    // helper function: get a vector from config that may be a scalar, a ConfigArray, or nested ConfigArrays meant to be flattened
    template <typename E>
    static vector<E> FlattenedVectorFrom(const ConfigValuePtr& valp)
    {
        if (valp.Is<vector<E>>())
            return valp.AsRef<vector<E>>(); // UNTESTED
        else if (valp.Is<ConfigArray>())
            return valp.AsRef<ConfigArray>().AsVector<E>([&](const wstring& msg) { valp.Fail(msg); }, /*flatten=*/true);
        else
            return std::vector<E>(1, (const E&)valp); // single element
    }
};
typedef shared_ptr<ConfigArray> ConfigArrayPtr;

// -----------------------------------------------------------------------
// ConfigLambda -- a lambda
// -----------------------------------------------------------------------

class ConfigLambda : public Object
{
public:
    typedef std::map<std::wstring, ConfigValuePtr> NamedParams; // TODO: maybe even not use a typedef, just use the type
private:
    // the function itself is a C++ lambda
    function<ConfigValuePtr(std::vector<ConfigValuePtr> &&, NamedParams &&, const std::wstring &exprName)> f;
    // inputs. This defines the interface to the function. Very simple in our case though.
    // We pass rvalue references because that allows to pass Thunks.
    std::vector<std::wstring> paramNames; // #parameters and parameter names (names are used for naming expressions only)
    NamedParams namedParams;              // lists named parameters with their default values. Named parameters are optional and thus always must have a default.
public:
    template <typename F>
    ConfigLambda(std::vector<std::wstring> &&paramNames, NamedParams &&namedParams, const F &f)
        : paramNames(move(paramNames)), namedParams(move(namedParams)), f(f)
    {
    }
    size_t GetNumParams() const
    {
        return paramNames.size();
    }
    const std::vector<std::wstring> &GetParamNames() const
    {
        return paramNames;
    } // used for expression naming
    // what this function does is call f() held in this object with the given arguments except optional arguments are verified and fall back to their defaults if not given
    // The arguments are rvalue references, which allows us to pass Thunks, which is important to allow stuff with circular references like CNTK's DelayedNode.
    ConfigValuePtr Apply(std::vector<ConfigValuePtr> &&args, NamedParams &&namedArgs, const std::wstring &exprName) const
    {
        NamedParams actualNamedArgs;
        // actualNamedArgs is a filtered version of namedArgs that contains all optional args listed in namedParams,
        // falling back to their default if not given in namedArgs.
        // On the other hand, any name in namedArgs that is not found in namedParams should be rejected.
        for (const auto &namedParam : namedParams)
        {
            const auto &id = namedParam.first;      // id of expected named parameter
            const auto valuei = namedArgs.find(id); // was such parameter passed?
            if (valuei == namedArgs.end())          // named parameter not passed
            {                                       // if not given then fall back to default
                auto f = [&namedParam]()            // we pass a lambda that resolves it upon first use, in our original location
                {
                    return namedParam.second.ResolveValue();
                };
                actualNamedArgs[id] = move(ConfigValuePtr::MakeThunk(f, namedParam.second.GetFailFn(), exprName));
            }
            else                                            // named parameter was passed
                actualNamedArgs[id] = move(valuei->second); // move it, possibly remaining unresolved
            // BUGBUG: we should pass in the location of the identifier, not that of the expression
        }
        for (const auto &namedArg : namedArgs) // make sure there are no extra named args that the macro does not take
            if (namedParams.find(namedArg.first) == namedParams.end())
                namedArg.second.Fail(L"function does not have an optional argument '" + namedArg.first + L"'");
        return f(move(args), move(actualNamedArgs), exprName);
    }
    // TODO: define an overload that takes const & for external users (which will then take a copy and pass it on to Apply &&)
};
typedef shared_ptr<ConfigLambda> ConfigLambdaPtr;

// -----------------------------------------------------------------------
// CustomConfigRecord -- helper for implementors of IConfigRecord
// Custom classes that implement IConfigRecord can derive from this to make
// it easier to manage the simulated config record.
// -----------------------------------------------------------------------

struct CustomConfigRecord : public IConfigRecord // any class that exposes config can derive from this
{
    const ConfigValuePtr& /*IConfigRecord::*/ operator[](const std::wstring& id) const override // e.g. confRec[L"message"]
    {
        const auto* valuep = Find(id);
        if (!valuep)
            RuntimeError("Unknown configuration-record member '%ls'", id.c_str());
        return *valuep;
    }

    const ConfigValuePtr* /*IConfigRecord::*/ Find(const std::wstring& id) const // returns nullptr if not found
    {
        const auto& mapIter = members.find(id);
        if (mapIter != members.end())
            return &mapIter->second;
        LazyCreateConfigMember(id);
        const auto& mapIter2 = members.find(id);
        if (mapIter2 != members.end())
            return &mapIter2->second;
        else
            return nullptr;
    }

    void InsertConfigMember(const std::wstring& id, ConfigValuePtr&& valuep) const/*because it is called from Find() which is const*/
    {
        const auto res = members.insert(make_pair(id, move(valuep)));
        assert(&res.first->second == &members.find(id)->second);
        assert(res.second);        // this says whether it has been inserted. It better be.
    }

    // call this whenever anything changes about this node
    // Once we use a ComputationNode as a config record, we are in immutable BS world.
    // So we should not really need to ever call this.
    // However, we *must* call it at least in DetachInputs() in order to break cyclic dependencies.
    void ClearConfigMemberCache()
    {
        members.clear();
    }

    // user of this class must implement LazyCreateConfigMember() and GetMemberIds()
    virtual void LazyCreateConfigMember(const std::wstring &id) const = 0;

protected:
    // cached return values from IConfigRecord implementation
    mutable std::map<std::wstring, ScriptableObjects::ConfigValuePtr> members; // [id] -> cached ConfigValuePtr
};

// -----------------------------------------------------------------------
// ConfigurableRuntimeType -- interface to scriptable runtime types
// -----------------------------------------------------------------------

// helper for configurableRuntimeTypes initializer below
// This returns a ConfigurableRuntimeType info structure that consists of
//  - a lambda that is a constructor for a given runtime type and
//  - a bool saying whether T derives from IConfigRecord
struct ConfigurableRuntimeType // TODO: rename to ScriptableObjects::Factory or something like that
{
    bool isConfigRecord; // exposes IConfigRecord  --in this case the expression name is computed differently, namely relative to this item
    // TODO: is this ^^ actually still used anywhere?
    function<shared_ptr<Object>(const IConfigRecordPtr)> construct; // lambda to construct an object of this class
    // TODO: we should pass the expression name to construct() as well
};

// scriptable runtime types must be exposed by this function
// TODO: should this be a static member of above class?
const ConfigurableRuntimeType *FindExternalRuntimeTypeInfo(const std::wstring &typeId);

// -----------------------------------------------------------------------
// ConfigurableRuntimeTypeRegister -- static table of all configurable runtime types
// -----------------------------------------------------------------------

class ConfigurableRuntimeTypeRegister
{
    // we wrap the static variable in a function so that we don't need a CPP file
    static std::map<std::wstring, ConfigurableRuntimeType> &GetTheRegister()
    {
        // the one static variable that contains all configurable runtime types
        static std::map<std::wstring, ConfigurableRuntimeType> reg;
        return reg;
    }

    static void Register(const wchar_t *typeId, ConfigurableRuntimeType &&rtInfo)
    {
        auto &reg = GetTheRegister();
        auto res = reg.insert(std::make_pair((std::wstring) typeId, std::move(rtInfo)));
        if (!res.second)
            LogicError("RegisterConfigurableRuntimeType: Attempted to register type '%ls' twice.", typeId);
    }

public:
    // to instantiate a ConfigurableRuntimeType object, use this function to find its constructor
    static const ConfigurableRuntimeType *Find(const std::wstring &typeId)
    {
        auto &reg = GetTheRegister();
        auto iter = reg.find(typeId);
        if (iter == reg.end())
            return nullptr;
        else
            return &iter->second;
    }

    // to register a runtime type, use an instance of this class in each library
    // ConfigurableRuntimeTypeRegister::Add<ClassName> registerClassName(L"ClassName")l
    template <class C>
    struct Add
    {
        Add(const wchar_t *typeId)
        {
            // create the runtime info
            ConfigurableRuntimeType rtInfo;
            rtInfo.construct = [](const IConfigRecordPtr config) // lambda to construct--this lambda can construct both the <float> and the <double> variant based on config parameter 'precision'
            {
                return MakeRuntimeObject<C>(config);
            };
            rtInfo.isConfigRecord = is_base_of<IConfigRecord, C>::value;
            // insert it into the static table
            Register(typeId, std::move(rtInfo));
        }
    };

    // to register a class that exists in dual precisions (Something<ElemType>>, use this one instead
    // ConfigurableRuntimeTypeRegister::AddFloatDouble<ClassName<float>,ClassName<double>> registerClassName(L"ClassName")l
    template <class Cfloat, class Cdouble>
    struct AddFloatDouble
    {
        AddFloatDouble(const wchar_t *typeId)
        {
            // create the runtime info
            ConfigurableRuntimeType rtInfo;
            rtInfo.construct = [](const IConfigRecordPtr config) // lambda to construct--this lambda can construct both the <float> and the <double> variant based on config parameter 'precision'
            {
                std::wstring precision = (*config)[L"precision"]; // dispatch on ElemType
                if (precision == L"float")
                    return MakeRuntimeObject<Cfloat>(config);
                else if (precision == L"double")
                    return MakeRuntimeObject<Cdouble>(config);
                else
                    RuntimeError("invalid value '%ls' for 'precision', must be 'float' or 'double'", precision.c_str());
            };
            rtInfo.isConfigRecord = is_base_of<IConfigRecord, Cfloat>::value;
            static_assert(is_base_of<IConfigRecord, Cfloat>::value == is_base_of<IConfigRecord, Cdouble>::value, ""); // we assume that both float and double have the same behavior
            // insert it into the static table
            Register(typeId, std::move(rtInfo));
        }
    };
};

// -----------------------------------------------------------------------
// IConfigRecord emulation of old CNTK config/NL
// This allows code written for CNTK config to simply turn ConfigParameters into a template parameter and accept an IConfigRecord.
// -----------------------------------------------------------------------

inline const ConfigValuePtr &IConfigRecord::operator()(const std::wstring &id) const
{
    return operator[](id);
} // e.g. confRec(L"message") instead of confRec[L"message"]
template <class T>
inline std::vector<T> IConfigRecord::operator()(const std::wstring &id, const std::vector<T> &defaultValue) const // retrieve an argvector (which derives from std::vector)
{
    const auto *valp = Find(id);
    if (!valp)
        return defaultValue; // default value
    if (!valp->Is<ConfigArray>())
        return std::vector<T>(1, (const T &) *valp); // scalar value
    const ConfigArray &arr = *valp;                  // actual array
#if 1                                                // TODO: test whether this works correctly w.r.t. typecasting
    return arr.AsVector<T>([&](const std::wstring &msg) { valp->Fail(msg); });
#else
    const auto size = arr.GetSize([&](const std::wstring &msg) { valp->Fail(msg); });
    std::vector<T> res(size);
    for (int i = 0; i < size; i++)
        res[i] = (const T &) arr.At(i);
    return res;
#endif
}
inline bool IConfigRecord::ExistsCurrent(const std::wstring &id) const // this is inefficient, but we can optimize it if it ever turns out to be a problem. I rather think, this function is misguided. The name is bad, too.
{
    for (const auto &idIter : GetMemberIds()) // linear scan. Not using STL algorithm to avoid pulling in a big header at this level
        if (idIter == id)
            return true;
    return false;
}
inline bool IConfigRecord::Match(const std::wstring &id, const std::wstring &compareValue) const
{
    auto *valp = Find(id);
    std::wstring val = valp ? *valp : std::wstring();
    return EqualCI(compareValue, val);
}
inline const std::string IConfigRecord::ConfigName() const
{
    LogicError("ConfigName not supported by BrainScript."); // needed in BinaryWriter
}
/*static*/ inline const IConfigRecord &IConfigRecord::Record() // empty record to be passed as a default to operator() when retrieving a nested ConfigRecord
{
    static struct EmptyConfigRecord : public IConfigRecord
    {
        virtual const ScriptableObjects::ConfigValuePtr &operator[](const std::wstring &) const override final
        {
            InvalidArgument("EmptyConfigRecord: Attempted to return a value from the empty record.");
        }
        virtual const ScriptableObjects::ConfigValuePtr *Find(const std::wstring &) const override final
        {
            return nullptr;
        }
        virtual std::vector<std::wstring> GetMemberIds() const
        {
            return std::vector<std::wstring>();
        }
    } emptyParameters;
    return emptyParameters;
}
template <class V>
/*static*/ const std::vector<typename V::value_type> &IConfigRecord::Array(const V &vec)
{
    return static_cast<const std::vector<typename V::value_type> &>(vec);
} // use this specifically for XXXargvector

}}} // end namespaces
