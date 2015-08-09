// ConfigRuntime.h -- execute what's given in a config file

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
    // To get a value of an expected type T, dynamic-cast that base pointer to Wrapped<T>.
    // Pointers to type U have the type shared_ptr<U>.

    struct ConfigValuePtr : public shared_ptr<Object>
    {
        bool currentlyResolving;    // set during resolution phase, to detect circular references
        TextLocation location;      // in source code
        template<typename T> Wrapped<T> * DynamicCastConfigValue() const {
            const auto p = get(); p;
            const auto r = dynamic_cast<Wrapped<T>*>(get());
            return r;
        }    // this casts the raw pointer that's inside the shared_ptr
    public:
        // construction     ---TODO: no template here
        template<typename T>
        ConfigValuePtr(const shared_ptr<T> & p, TextLocation location) : shared_ptr<Object>(p), currentlyResolving(false), location(location) {}
        ConfigValuePtr() : currentlyResolving(false) {} // (formally needed somehow)
        // methods for retrieving values
        // One accesses when values are constant, so we can just return values as const &.
        operator double()  const { return AsConfigValue<double>(); }
        operator wstring() const { return AsConfigValue<wstring>(); }
        operator bool()    const { return AsConfigValue<bool>(); }
        template<typename T> operator shared_ptr<T>() const { return AsConfigValue<shared_ptr<T>>(); }
        operator size_t() const
        {
            const auto val = AsConfigValue<double>();
            const auto ival = (size_t)val;
            if (ival != val)
                throw EvaluationError(L"numeric value is not an integer", location);
            // TODO: ^^this cannot be done, since we don't have TextLocation here.
            return ival;
        }
        // type helpers
        template<typename T> bool IsConfigValue() const { return DynamicCastConfigValue<T>() != nullptr; }
        template<typename T> T & AsConfigValue() const     // returns reference to what the 'value' member
        {
            auto * p = DynamicCastConfigValue<T>();        // -> Wrapped<T>
            if (p == nullptr)   // TODO: can we make this look the same as TypeExpected in ConfigRuntime.cpp? We'd need the type name
                throw EvaluationError(L"config member has wrong type", location);
            return p->value;                    // this unwraps the value out from its Wrapped wrapper
        }
        // TODO: clean this up; get rid of specalization
        template<> bool IsConfigValue<wstring>() const
        {
            const auto p = dynamic_cast<wstring*>(get());
            return p != nullptr;
        }
        template<> wstring & AsConfigValue<wstring>() const     // returns reference to what the 'value' member
        {
            const auto p = dynamic_cast<wstring*>(get());
            if (p == nullptr)   // TODO: can we make this look the same as TypeExpected in ConfigRuntime.cpp? We'd need the type name
                throw EvaluationError(L"config member has wrong type", location);
            return *p;
        }
        const char * TypeName() const { return typeid(*get()).name(); }
        // methods for resolving the value
        template<typename F>
        void ResolveValue(const F & Evaluate, TextLocation location)
        {
            // call this when a a member might be as-of-yet unresolved, to evaluate it on-demand
            // value.get() is a pointer to Wrapped<type of value>
            // Type of value is ExpressionPtr if the value is not yet resolved.
            auto * p = DynamicCastConfigValue<ExpressionPtr>();    // -> Wrapped<ExpressionPtr>
            if (!p)                             // value is not an ExpressionPtr: we already got a proper value; done.
                return;
            if (currentlyResolving)             // detect circular references (infinite recursion)
                throw EvaluationError(L"circular reference (expression to compute identifier's value uses the identifier's value)", location);
            currentlyResolving = true;
            ExpressionPtr valueExpr = p->value;
            *this = Evaluate(valueExpr);        // completely replace ourselves with the actual result
            if (currentlyResolving)
                LogicError("ResolveValue: spurious 'currentlyResolving' flag");
        }
        // resolution
        template<typename F>
        void ResolveValue(const F & Evaluate)
        {
            ConfigValuePtr::ResolveValue(Evaluate, location);
        }
    };

    template<typename T> static inline ConfigValuePtr MakeConfigValue(const T & val, TextLocation location) { return ConfigValuePtr(make_shared<Wrapped<T>>(val), location); }
    // strings are stored in a String instead
    template<> ConfigValuePtr static inline MakeConfigValue<wstring>(const wstring & val, TextLocation location) {
        const auto r = ConfigValuePtr(make_shared<String>(val), location);
        return r;
    }

    class ConfigRecord      // all configuration arguments to class construction, resolved into ConfigValuePtrs
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
        template<typename F>
        void ResolveAll(const F & Evaluate)   // resolve all members; do this before handing a ConfigRecord to C++ code
        {
            for (auto & member : members)
                member.second.ResolveValue(Evaluate);
        }
    };
    typedef shared_ptr<ConfigRecord> ConfigRecordPtr;       // dictionaries evaluate to this

    // an array is just a vector of config values; like ConfigRecord, it can be wrapped as a value in a Wrapped
    typedef vector<ConfigValuePtr> ConfigArray;  // TODO: change to vector<ConfigValuePtr>

    // understand and execute from the syntactic expression tree
    ConfigValuePtr Evaluate(ExpressionPtr);     // evaluate the expression tree
    void Do(ExpressionPtr e);                   // evaluate e.do

}}} // end namespaces
