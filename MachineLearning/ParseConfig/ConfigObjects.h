// ConfigObjects.h -- objects that the config parser operates on

#pragma once

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;

    struct Polymorphic { virtual ~Polymorphic() { } };

    // TODO: a ConfigValuePtr should be a shared_ptr to the value directly (such as ComputationNode), while having the base class
    // ConfigValues are value structs. E.g. we can copy them to construct a ConfigValuePtrfrom them.
    template<typename T> class ConfigValue : public Polymorphic
    {
    public:
        /*const*/ T value;      // primitive type (e.g. double) or shared_ptr<runtime type>
        ConfigValue(T value) : value(value) { } // TODO: take a shared_ptr<T> and construct base shared_ptr from it
    };

    // a string (STL wstring, to be precise) that can be help in a ConfigValuePtr
    // TODO: templatize this, call it ConfigObject
    class ConfigString : public Polymorphic, public wstring
    {
    public:
        ConfigString(const wstring & val) : wstring(val) { }
    };

}}} // end namespaces
