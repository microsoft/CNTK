// ConfigRuntime.h -- execute what's given in a config file

#pragma once

#include "Basics.h"
#include "ParseConfig.h"
#include <memory>   // for shared_ptr

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;

    class EvaluationError : public ConfigError
    {
    public:
        EvaluationError(const wstring & msg, TextLocation where) : ConfigError(msg, where) { }
        /*implement*/ const wchar_t * kind() const { return L"evaluating"; }
    };

    // config values
    // All values in a ConfigRecord derive from ConfigValueBase.
    // To get a value of an expected type T, dynamic-cast that base pointer to ConfigValue<T>.
    // Pointers to type U have the type shared_ptr<U>.

    struct Polymorphic { virtual ~Polymorphic() { } };

    // TODO: this goes elsewhere
    struct ConfigValueBase { virtual ~ConfigValueBase(){} };    // one value in a config dictionary

    // TODO: a ConfigValuePtr should be a shared_ptr to the value directly (such as ComputationNode), while having the base class
    // ConfigValues are value structs. E.g. we can copy them to construct a ConfigMember from them.
    template<typename T> class ConfigValue : public ConfigValueBase
    {
    public:
        /*const*/ T value;      // primitive type (e.g. double) or shared_ptr<runtime type>
        ConfigValue(T value) : value(value) { } // TODO: take a shared_ptr<T> and construct base shared_ptr from it
    };

    struct ConfigValuePtr : public shared_ptr<ConfigValueBase>
    {
        bool currentlyResolving;    // set during resolution phase, to detect circular references
        TextLocation location;      // in source code
    public:
        // construction     ---TODO: no template here
        template<typename T>
        ConfigValuePtr(const shared_ptr<T> & p, TextLocation location) : shared_ptr<ConfigValueBase>(p), currentlyResolving(false), location(location) {}
        ConfigValuePtr() : currentlyResolving(false) {} // (formally needed somehow)
        // accessing values
        // One accesses when values are constant, so we can just return values as const &.
        template<typename T> ConfigValue<T> * DynamicCast() const { return dynamic_cast<ConfigValue<T>*>(get()); }    // this casts the raw pointer that's inside the shared_ptr
        template<typename T> bool Is() const { return DynamicCast<T>() != nullptr; }
        //template<typename T> bool Is() const { return dynamic_cast<ConfigValue<T>*>(get()) != nullptr; }  // test for type
        template<typename T> T & As() const     // returns reference to what the 'value' member
        {
            auto * p = DynamicCast<T>();        // -> ConfigValue<T>
            if (p == nullptr)   // TODO: can we make this look the same as TypeExpected in ConfigRuntime.cpp? We'd need the type name
                throw EvaluationError(L"config member has wrong type", TextLocation()); // TODO: we need location here
            return p->value;                    // this unwraps the value out from its ConfigValue wrapper
        }
        operator double() const { return As<double>(); }
        operator wstring() const { return As<wstring>(); }
        operator bool() const { return As<bool>(); }
        operator size_t() const
        {
            const auto val = As<double>();
            const auto ival = (size_t)val;
            if (ival != val)
                throw EvaluationError(L"numeric value is not an integer", TextLocation());
            // TODO: ^^this cannot be done, since we don't have TextLocation here.
            return (size_t)As<double>();
        }
        // methods for retrieving values
        template<typename T> operator shared_ptr<T>() const { return As<shared_ptr<T>>(); }
        //operator ConfigValuePtr() const { return value; }   // or the untyped config value
        // resolving
        // methods for resolving the value
        template<typename F>
        void ResolveValue(const F & Evaluate, TextLocation location)
        {
            // call this when a a member might be as-of-yet unresolved, to evaluate it on-demand
            // value.get() is a pointer to ConfigValue<type of value>
            // Type of value is ExpressionPtr if the value is not yet resolved.
            auto * p = DynamicCast<ExpressionPtr>();    // -> ConfigValue<ExpressionPtr>
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


        const char * TypeName() const { return typeid(*get()).name(); }
        // resolution
        template<typename F>
        void ResolveValue(const F & Evaluate)
        {
            ConfigValuePtr::ResolveValue(Evaluate, location);
        }
    };

    template<typename T> ConfigValuePtr MakeConfigValue(const T & val, TextLocation location) { return ConfigValuePtr(make_shared<ConfigValue<T>>(val), location); }

    class ConfigRecord      // all configuration arguments to class construction, resolved into ConfigValuePtrs
    {
    public:
        typedef ConfigValuePtr ConfigMember;
    private:
        map<wstring, ConfigMember> members;
    public:
        // regular lookup: just use record[id]
        const ConfigMember & operator[](const wstring & id) const // e.g. confRec[L"message"]
        {
            const auto memberIter = members.find(id);
            if (memberIter == members.end())
                RuntimeError("unknown class parameter");
            return memberIter->second;
        }
        ConfigMember * Find(const wstring & id)                 // returns nullptr if not found
        {
            auto memberIter = members.find(id);
            if (memberIter == members.end())
                return nullptr;
            else
                return &memberIter->second;
        }
        bool empty() const { return members.empty(); }      // late-init object constructors can test this
        // add a member
        void Add(const wstring & id, TextLocation idLocation, ConfigValuePtr value) { members[id] = ConfigMember(value, idLocation); }
        // member resolution
        template<typename F>
        void ResolveAll(const F & Evaluate)   // resolve all members; do this before handing a ConfigRecord to C++ code
        {
            for (auto & member : members)
                member.second.ResolveValue(Evaluate);
        }
    };
    typedef shared_ptr<ConfigRecord> ConfigRecordPtr;       // dictionaries evaluate to this

    // an array is just a vector of config values; like ConfigRecord, it can be wrapped as a value in a ConfigValue
    typedef vector<ConfigValuePtr> ConfigArray;  // TODO: change to vector<ConfigMember>

    // understand and execute from the syntactic expression tree
    ConfigValuePtr Evaluate(ExpressionPtr);

}}} // end namespaces
