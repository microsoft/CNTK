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
    struct ConfigValuePtr : public shared_ptr<ConfigValueBase>
    {
        template<typename T>
        ConfigValuePtr(const shared_ptr<T> & val) : shared_ptr<ConfigValueBase>(val){}
        ConfigValuePtr(){}
    };

    // TODO: a ConfigValuePtr should be a shared_ptr to the value directly (such as ComputationNode), while having the base class
    template<typename T> class ConfigValue : public ConfigValueBase
    {
    public:
        // TODO: derive this from shared_ptr<T>, where 
        /*const*/ T value;      // primitive type (e.g. double) or shared_ptr<runtime type>
        ConfigValue(T value) : value(value) { } // TODO: take a shared_ptr<T> and construct base shared_ptr from it
    };

    template<typename T> ConfigValuePtr MakeConfigValue(const T & val) { return make_shared<ConfigValue<T>>(val); }

    class ConfigRecord      // all configuration arguments to class construction, resolved into ConfigValuePtrs
    {
    public:
        class ConfigMember  // TODO: can a ConfigMember not just be a ConfigValuePtr with conversion functions? and get rid of 'value'
        {
            // TODO: got a double shared_ptr here. Instead,
            // wrap constants into objects as well
            ConfigValuePtr value;       // ... TODO: ConfigValues can be passed around by value
            bool currentlyResolving;    // set during resolution phase, to detect circular references
            TextLocation location;      // in source code  --TODO: initialize this to some meaningful value
            template<typename T> T * As() const
            {
                auto * p = dynamic_cast<T*>(value.get());
                if (p == nullptr)   // TODO: can we make this look the same as TypeExpected in ConfigRuntime.cpp? We'd need the type name
                    throw EvaluationError(L"config member has wrong type", location);
                return p;
            }
        public:
            // methods for retrieving values
            operator double() const { return As<ConfigValue<double>>()->value; }
            operator wstring() const { return As<ConfigValue<wstring>>()->value; }
            operator bool() const { return As<ConfigValue<bool>>()->value; }
            operator size_t() const { return (size_t) As<ConfigValue<double>>()->value; }   // TODO: fail if fractional
            template<typename T> operator shared_ptr<T>() const { return As<ConfigValue<shared_ptr<T>>>()->value; }
            operator ConfigValuePtr() const { return value; }   // or the untyped config value
            template<typename T> bool Is() const { return dynamic_cast<ConfigValue<T>*>(value.get()) != nullptr; }  // test for type
            // BUGBUG: ^^ does not work for testing if type is derived from T
            const char * TypeName() const { return typeid(*value.get()).name(); }
            // methods for resolving the value
            template<typename F>
            void ResolveValue(const F & Evaluate)
            {
                // call this when a a member might be as-of-yet unresolved, to evaluate it on-demand
                // value.get() is a pointer to ConfigValue<type of value>
                // Type of value is ExpressionPtr if the value is not yet resolved.
                auto * p = dynamic_cast<ConfigValue<ExpressionPtr>*>(value.get());
                if (!p)                             // value is not an ExpressionPtr: we already got a proper value; done.
                    return;
                const auto valueExpr = p->value;
                if (currentlyResolving)             // detect circular references (infinite recursion)
                    throw EvaluationError(L"circular reference (expression to compute identifier's value uses the identifier's value)", location);
                currentlyResolving = true;
                value = Evaluate(valueExpr);        // evaluate and replace 'value' with real value
                currentlyResolving = false;
            }
            // constructors
            ConfigMember(ConfigValuePtr value, TextLocation location) : value(value), currentlyResolving(false), location(location) {}
            ConfigMember() : currentlyResolving(false) {}    // needed for map below
        };
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
