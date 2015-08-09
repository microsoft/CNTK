// ConfigEvaluator.h -- execute what's given in a config file

#pragma once

#include "Basics.h"
#include "ConfigParser.h"
#include "ConfigObjects.h"
#include <memory>   // for shared_ptr

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;

    // error object

    class EvaluationError : public ConfigError
    {
    public:
        EvaluationError(const wstring & msg, TextLocation where) : ConfigError(msg, where) { }
        /*implement*/ const wchar_t * kind() const { return L"evaluating"; }
    };

    // config values
    // All values in a ConfigRecord derive from Object.
    // To get a value of an expected type T, dynamic-cast that base pointer to BoxOfWrapped<T>.
    // Pointers to type U have the type shared_ptr<U>.

    struct ConfigValuePtr : public shared_ptr<Object>
    {
        TextLocation location;      // in source code
        template<typename T> BoxOfWrapped<T> * DynamicCastBoxOfWrapped() const {
            const auto p = get(); p;
            const auto r = dynamic_cast<BoxOfWrapped<T>*>(get());
            return r;
        }    // this casts the raw pointer that's inside the shared_ptr
    public:
        // construction     ---TODO: no template here
        template<typename T>
        //ConfigValuePtr(const shared_ptr<T> & p, TextLocation location) : shared_ptr<Object>(dynamic_pointer_cast<Object>(p)), location(location) {}
        ConfigValuePtr(const shared_ptr<T> & p, TextLocation location) : shared_ptr<Object>(p), location(location) {}
        ConfigValuePtr() {} // (formally needed somehow)
        // methods for retrieving values
        // One accesses when values are constant, so we can just return values as const &.
        //operator double() const { return AsBoxOfWrapped<double>(); } DELETE THIS when fully tested
        //operator bool()   const { return AsBoxOfWrapped<bool>(); }
        operator double() const { return (Double)*this; }
        operator bool() const { return (Bool)*this; }
        template<typename T> operator T() const { return As<T>(); }
        operator size_t() const
        {
            const auto val = AsBoxOfWrapped<double>();
            const auto ival = (size_t)val;
            if (ival != val)
                throw EvaluationError(L"numeric value is not an integer", location);
            // TODO: ^^this cannot be done, since we don't have TextLocation here.
            return ival;
        }
        // type helpers
        template<typename T> bool IsBoxOfWrapped() const { return DynamicCastBoxOfWrapped<T>() != nullptr; }
        template<typename T> T & AsBoxOfWrapped() const     // returns reference to what the 'value' member
        {
            auto * p = DynamicCastBoxOfWrapped<T>();        // -> BoxOfWrapped<T>
            if (p == nullptr)   // TODO: can we make this look the same as TypeExpected in ConfigRuntime.cpp? We'd need the type name
                throw EvaluationError(L"config member has wrong type", location);
            return *p;                    // this unwraps the value out from its BoxOfWrapped wrapper
        }
        // TODO: clean this up; get rid of specalization
        template<class C>
        bool Is() const
        {
            const auto p = dynamic_cast<C*>(get());
            return p != nullptr;
        }
        template<class C>
        C & As() const     // returns reference to what the 'value' member
        {
            const auto p = dynamic_cast<C*>(get());
            if (p == nullptr)   // TODO: can we make this look the same as TypeExpected in ConfigRuntime.cpp? We'd need the type name
                throw EvaluationError(L"config member has wrong type", location);
            return *p;
        }
        template<class C>
        shared_ptr<C> AsPtr() const     // returns a shared_ptr cast to the 'value' member
        {
            const auto p = dynamic_pointer_cast<C>(*this);
            if (!p)             // TODO: can we make this look the same as TypeExpected in ConfigRuntime.cpp? We'd need the type name
                throw EvaluationError(L"config member has wrong type", location);
            return p;
        }
        const char * TypeName() const { return typeid(*get()).name(); }
        // methods for resolving the value
        // Thunk for resolving a value. This Object represents a function that returns a ConfigValuePtr; call to resolve a deferred value
        class Thunk : public Object
        {
            function<ConfigValuePtr()> f;   // the function to compute the value
            bool currentlyResolving;        // set during resolution phase, to detect circular references
            TextLocation location;          // in source code
        public:
            Thunk(function<ConfigValuePtr()> f, TextLocation location) : f(f), location(location), currentlyResolving(false) { }
            ConfigValuePtr ResolveValue()
            {
                if (currentlyResolving)                 // detect circular references (infinite recursion)
                    throw EvaluationError(L"circular reference (expression to compute identifier's value uses the identifier's value)", location);
                currentlyResolving = true;              // can't run from inside ourselves
                return f();
                // no need to reset currentlyResolving because this object gets replaced anyway
            }
        };
        void ResolveValue()
        {
            // call this when a a member might be as-of-yet unresolved, to evaluate it on-demand
            // get() is a pointer to a Thunk in that case, that is, a function object that yields the value
            const auto thunkp = dynamic_cast<Thunk*>(get());   // is it a Thunk?
            if (!thunkp)                            // value is not a Thunk: we already got a proper value; done.
                return;
            *this = thunkp->ResolveValue();         // completely replace ourselves with the actual result. This also releases the Thunk object
            ResolveValue();                         // allow it to return another Thunk...
        }
    };

    template<typename T> ConfigValuePtr static inline MakeBoxedConfigValue(const T & val, TextLocation location) {
        const auto r = ConfigValuePtr(make_shared<T>(val), location);
        return r;
    }
    // use this for old-style classes, TO BE REMOVED
    template<typename T> static inline ConfigValuePtr MakeWrappedAndBoxedConfigValue(const T & val, TextLocation location) {
        return ConfigValuePtr(make_shared<BoxOfWrapped<T>>(val), location);
    }
    // use this for primitive values, double and bool
    template<typename T> static inline ConfigValuePtr MakePrimitiveConfigValue(const T & val, TextLocation location) {
        return MakeWrappedAndBoxedConfigValue(val, location);
    }
    // strings are stored in a String instead
    ConfigValuePtr static inline MakeStringConfigValue(const String & val, TextLocation location) {
        return MakeBoxedConfigValue(val, location);
    }

    class ConfigRecord : public Object      // all configuration arguments to class construction, resolved into ConfigValuePtrs
    {
        map<wstring, ConfigValuePtr> members;
    public:
        // regular lookup: just use record[id]
        const ConfigValuePtr & operator[](const wstring & id) const // e.g. confRec[L"message"]
        {
            const auto memberIter = members.find(id);
            if (memberIter == members.end())
                RuntimeError("unknown class parameter");
            return memberIter->second;
        }
        ConfigValuePtr * Find(const wstring & id)                 // returns nullptr if not found
        {
            auto memberIter = members.find(id);
            if (memberIter == members.end())
                return nullptr;
            else
                return &memberIter->second;
        }
        bool empty() const { return members.empty(); }      // late-init object constructors can test this
        // add a member
        void Add(const wstring & id, TextLocation idLocation, ConfigValuePtr value) { members[id] = ConfigValuePtr(value, idLocation); }
        // member resolution
        void ResolveAll()   // resolve all members; do this before handing a ConfigRecord to C++ code
        {
            for (auto & member : members)
                member.second.ResolveValue();
        }
    };
    typedef shared_ptr<ConfigRecord> ConfigRecordPtr;       // dictionaries evaluate to this

    // an array is just a vector of config values; like ConfigRecord, it can be wrapped as a value in a BoxOfWrappedWrapped
    typedef vector<ConfigValuePtr> ConfigArray;  // TODO: change to vector<ConfigValuePtr>

    // understand and execute from the syntactic expression tree
    ConfigValuePtr Evaluate(ExpressionPtr);     // evaluate the expression tree
    void Do(ExpressionPtr e);                   // evaluate e.do

}}} // end namespaces
