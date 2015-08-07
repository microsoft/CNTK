// ConfigRuntime.h -- execute what's given in a config file

#pragma once

#include "Basics.h"
#include "ConfigurableRuntimeObjects.h"
#include "ParseConfig.h"
#include <memory>   // for shared_ptr

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;

    class EvaluationError : public ConfigError
    {
    public:
        EvaluationError(const wstring & msg, TextLocation where) : ConfigError(msra::strfun::utf8(msg), where) { }
        /*implement*/ const char * kind() const { return "evaluating"; }
    };

    // config values
    // All values in a ConfigRecord derive from ConfigValueBase.
    // To get a value of an expected type T, dynamic-cast that base pointer to ConfigValue<T>.
    // Pointers to type U have the type shared_ptr<U>.

    // TODO: this goes elsewhere
    struct ConfigValueBase { virtual ~ConfigValueBase(){} };    // one value in a config dictionary
    typedef shared_ptr<ConfigValueBase> ConfigValuePtr;

    template<typename T> class ConfigValue : public ConfigValueBase
    {
    public:
        /*const*/ T value;      // primitive type (e.g. double) or shared_ptr<runtime type>
        ConfigValue(T value) : value(value) { }
    };

    class ConfigRecord      // all configuration arguments to class construction, resolved into ConfigValuePtrs
    {
    public:
        class ConfigMember
        {
            ConfigValuePtr value;
            TextLocation location;      // in source code
            template<typename T> T * As() const
            {
                auto * p = dynamic_cast<T*>(value.get());
                if (p == nullptr)   // TODO: can we make this look the same as TypeExpected in ConfigRuntime.cpp? We'd need the type name
                    throw EvaluationError(L"config member has wrong type", location);
                return p;
            }
        public:
            operator double() const { return As<ConfigValue<double>>()->value; }
            operator wstring() const { return As<ConfigValue<wstring>>()->value; }
            operator bool() const { return As<ConfigValue<bool>>()->value; }
            operator size_t() const { return (size_t) As<ConfigValue<double>>()->value; }   // TODO: fail if fractional
            template<typename T> operator shared_ptr<T>() const { return As<ConfigValue<shared_ptr<T>>>()->value; }
            ConfigMember(ConfigValuePtr value) : value(value) { }   // TODO: get the TextLocation as an arg in here & remember it
            ConfigMember(){}    // needed for map below
        };
    private:
        map<wstring, ConfigMember> members;
    public:
        const ConfigMember & operator[](const wstring & id) const // e.g. confRec[L"message"]
        {
            const auto memberIter = members.find(id);
            if (memberIter == members.end())
                RuntimeError("unknown class parameter");
            return memberIter->second;
        }
        void Add(const wstring & id, ConfigValuePtr value) { members[id] = ConfigMember(value); }
        bool empty() const { return members.empty(); }
    };

    // understand and execute from the syntactic expression tree
    ConfigValuePtr Evaluate(ExpressionPtr);

}}} // end namespaces
