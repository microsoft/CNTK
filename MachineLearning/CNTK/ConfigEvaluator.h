// ConfigEvaluator.h -- execute what's given in a config file

#pragma once

#include "Basics.h"
#include "ConfigParser.h"
#include "ConfigObjects.h"
#include <memory>   // for shared_ptr

namespace Microsoft{ namespace MSR { namespace CNTK { namespace Config {

    using namespace std;
    using namespace msra::strfun;   // for wstrprintf()

    // error object

    class EvaluationError : public ConfigError
    {
    public:
        EvaluationError(const wstring & msg, TextLocation where) : ConfigError(msg, where) { }
        /*Configerror::*/ const wchar_t * kind() const { return L"evaluating"; }
    };

    // config values
    // A ConfigValuePtr is a shared_ptr to something that derives from Object.
    // To get a shared_ptr<T> of an expected type T, type-cast the ConfigValuePtr to it.
    // To get the value of a copyable type like T=double or wstring, type-cast to T directly.

    class ConfigValuePtr : public shared_ptr<Object>
    {
        TextLocation location;      // in source code
        template<typename T> T * DynamicCast() const
        {
            ResolveValue();
            return dynamic_cast<T*>(get());
        }    // this casts the raw pointer that's inside the shared_ptr
        //void operator=(const ConfigValuePtr &);
        // TODO: copying ConfigValuePtrs if they are not resolved yet, as it may lead to multiple executions of the Thunk.
        //       Solve by either forbidding assignment (move only) or by resolving upon assignment and deal with the fallout.
        //       This is a little nasty.
        wstring expressionName;     // the name reflects the path to reach this expression in the (possibly dynamically macro-expanded) expression tree
    public:
        // construction     ---TODO: no template here
        template<typename T>
        ConfigValuePtr(const shared_ptr<T> & p, TextLocation location, const wstring & expressionName) : shared_ptr<Object>(p), location(location), expressionName(expressionName) { }
        ConfigValuePtr() {} // (formally needed somehow)
        // methods for retrieving values
        // access as a reference, that is, as a shared_ptr<T>   --use this for Objects
        template<typename T> operator shared_ptr<T>() const { return AsPtr<T>(); }
        // access as a (const & to) value  --use this for primitive types (also works to get a const wstring & from a String)
        template<typename T> operator T() const { return AsRef<T>(); }
        operator double() const { return AsRef<Double>(); }
        operator bool() const { return AsRef<Bool>(); }
        template<typename INT> INT AsInt() const
        {
            double val = AsRef<Double>();
            INT ival = (INT)val;
            const wchar_t * type = L"size_t";
            const char * t = typeid(INT).name(); t;
            // TODO: there is some duplication of type checking; can we unify that?
            if (ival != val)
                throw EvaluationError(wstrprintf(L"expected expression of type %ls instead of floating-point value %f", type, val), location);
            return ival;
        }
        operator size_t() const { return AsInt<size_t>(); }
        operator int() const { return AsInt<int>(); }
        // type helpers
        template<class C>
        bool Is() const
        {
            ResolveValue();
            const auto p = dynamic_cast<C*>(get());
            return p != nullptr;
        }
        template<class C>
        const C & AsRef() const     // returns reference to what the 'value' member. Configs are considered immutable, so return a const&
        {
            // Note: since this returns a reference into 'this', keep the object you call this on around as long as you use the returned reference!
            ResolveValue();
            const C * wanted = (C *) nullptr; const auto * got = get(); wanted; got;   // allows to see C in the debugger
            const auto p = dynamic_cast<C*>(get());
            if (p == nullptr)   // TODO: can we make this look the same as TypeExpected in ConfigEvaluator.cpp? We'd need the type name
                throw EvaluationError(L"config member has wrong type, expected a " + TypeId<C>(), location);
            return *p;
        }
        template<class C>
        shared_ptr<C> AsPtr() const     // returns a shared_ptr cast to the 'value' member
        {
            ResolveValue();
            const auto p = dynamic_pointer_cast<C>(*this);
            if (!p)             // TODO: can we make this look the same as TypeExpected in ConfigEvaluator.cpp? We'd need the type name
                throw EvaluationError(L"config member has wrong type, expected a " + TypeId<C>(), location);
            return p;
        }
        // properties
        const char * TypeName() const { return typeid(*get()).name(); }
        TextLocation GetLocation() const { return location; }
        const wstring & GetExpressionName() const{ return expressionName;  }
        // TODO: ^^ it seems by saving the name in the ConfigValuePtr itself, we don't gain anything; maybe remove again in the future
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
        void ResolveValue() const   // (this is const but mutates the value if it resolves)
        {
            // call this when a a member might be as-of-yet unresolved, to evaluate it on-demand
            // get() is a pointer to a Thunk in that case, that is, a function object that yields the value
            const auto thunkp = dynamic_cast<Thunk*>(get());   // is it a Thunk?
            if (!thunkp)                            // value is not a Thunk: we already got a proper value; done.
                return;
            const auto value = thunkp->ResolveValue();         // completely replace ourselves with the actual result. This also releases the Thunk object
            const_cast<ConfigValuePtr&>(*this) = value;
            ResolveValue();                         // allow it to return another Thunk...
        }
    };  // ConfigValuePtr

    // use this for primitive values, double and bool
    template<typename T> static inline ConfigValuePtr MakePrimitiveConfigValuePtr(const T & val, TextLocation location, const wstring & exprPath)
    {
        return ConfigValuePtr(make_shared<BoxOf<Wrapped<T>>>(val), location, exprPath);
    }

    // -----------------------------------------------------------------------
    // ConfigRecord -- collection of named config values
    // -----------------------------------------------------------------------

    struct IsConfigRecord   // any class that exposes config can derive from this
    {
        virtual const ConfigValuePtr & operator()(const wstring & id, wstring message = L"") const = 0; // e.g. config(L"arg", L"arg is the argument to this function")
        virtual const ConfigValuePtr & operator[](const wstring & id) const { return operator()(id); }  // e.g. confRec[L"message"]
        virtual const ConfigValuePtr * Find(const wstring & id) const = 0;                              // returns nullptr if not found
    };

    class ConfigRecord : public Object, public IsConfigRecord      // all configuration arguments to class construction, resolved into ConfigValuePtrs
    {
    public:
        typedef shared_ptr<ConfigRecord> ConfigRecordPtr;
    private:
        // change to ContextInsensitiveMap<ConfigValuePtr>
        map<wstring, ConfigValuePtr> members;
        ConfigRecordPtr parentRecord;           // we look up the chain
    public:

        // regular lookup: just use record[id]
        /*IsConfigRecord::*/ const ConfigValuePtr & operator()(const wstring & id, wstring message) const   // e.g. confRec(L"name", L"This specifies the object's internal name.")
        {
            const auto memberIter = members.find(id);
            if (memberIter != members.end())
                return memberIter->second;          // found
            if (parentRecord)
                return (*parentRecord)[id];         // not found but have parent: look it up there
            // failed: shown an error
            if (message.empty())
                throw EvaluationError(L"required parameter '" + id + L"' not found", TextLocation());
            else
                throw EvaluationError(L"required parameter '" + id + L"' not found. " + message, TextLocation());
        }
        /*IsConfigRecord::*/ const ConfigValuePtr * Find(const wstring & id) const         // returns nullptr if not found
        {
            auto memberIter = members.find(id);
            if (memberIter == members.end())
                if (parentRecord)
                    return parentRecord->Find(id);
                else
                    return nullptr;
            else
                return &memberIter->second;
        }
        bool empty() const { return members.empty(); }      // late-init object constructors can test this
        // add a member
        void Add(const wstring & id, TextLocation idLocation, ConfigValuePtr value) { members[id] = value; idLocation; }
        // TODO: ^^ idLocation is meant to hold the text location of the identifier
        // get members; used for optional argument lookup and logging
        const map<wstring, ConfigValuePtr> & GetMembers() const { return members; }
        // member resolution
        void ResolveAll()   // resolve all members; do this before handing a ConfigRecord to C++ code
        {
            for (auto & member : members)
                member.second.ResolveValue();
        }
    };
    typedef ConfigRecord::ConfigRecordPtr ConfigRecordPtr;

    // create a runtime object from its type --general case
    // There can be specializations of this that instantiate objects that do not take ConfigRecords or involve mapping like ComputationNode.
    template<typename C>
    shared_ptr<C> MakeRuntimeObject(const ConfigRecord & config)
    {
        return make_shared<C>(config);
    }

    // -----------------------------------------------------------------------
    // ConfigArray -- an array of config values
    // -----------------------------------------------------------------------

    // an array is just a vector of config values
    class ConfigArray : public Object
    {
        vector<ConfigValuePtr> values;
        int firstIndex;
        ConfigValuePtr & GetElem(int index, TextLocation indexLocation)
        {
            if (index < firstIndex || index >= firstIndex + values.size())
                throw EvaluationError(L"index out of bounds", indexLocation);
            return values[(size_t)(index - firstIndex)];
        }
    public:
        ConfigArray() : firstIndex(0) { }
        ConfigArray(int firstIndex, vector<ConfigValuePtr> && values) : firstIndex(firstIndex), values(values) { }
        pair<int, int> GetRange() const { return make_pair(firstIndex, firstIndex+(int)values.size()-1); }
        // building the array from expressions: append an element or an array
        void Append(ConfigValuePtr value) { values.push_back(value); }
        void Append(const ConfigArray & other) { values.insert(values.end(), other.values.begin(), other.values.end()); }
        // get element at index, including bounds check
        ConfigValuePtr At(int index, TextLocation indexLocation) /*const*/
        {
            auto & elem = GetElem(index, indexLocation);
            elem.ResolveValue();
            return elem;
        }
    };
    typedef shared_ptr<ConfigArray> ConfigArrayPtr;

    // -----------------------------------------------------------------------
    // ConfigLambda -- a lambda
    // -----------------------------------------------------------------------

    class ConfigLambda : public Object
    {
        // the function itself is a C++ lambda
        function<ConfigValuePtr(const vector<ConfigValuePtr>&, shared_ptr<ConfigRecord>, const wstring & exprName)> f;
        // inputs. This defines the interface to the function. Very simple in our case though.
        vector<wstring> paramNames;             // #parameters and parameter names (names are used for naming expressions only)
        shared_ptr<ConfigRecord> namedParams;   // lists named parameters with their default values. Named parameters are optional and thus always must have a default.
    public:
        template<typename F>
        ConfigLambda(const vector<wstring> & paramNames, shared_ptr<ConfigRecord> namedParams, const F & f) : paramNames(paramNames), namedParams(namedParams), f(f) { }
        size_t GetNumParams() const { return paramNames.size(); }
        const vector<wstring> & GetParamNames() const { return paramNames; }    // used for expression naming
        ConfigValuePtr Apply(vector<ConfigValuePtr> args, shared_ptr<ConfigRecord> namedArgs, const wstring & exprName)
        {
            auto actualNamedArgs = make_shared<ConfigRecord>();
            // actualNamedArgs is a filtered version of namedArgs that contains all optional args listed in namedParams,
            // falling back to their default if not given in namedArgs.
            // On the other hand, any name in namedArgs that is not found in namedParams should be rejected.
            for (const auto & namedParam : namedParams->GetMembers())
            {
                const auto & id = namedParam.first;                         // id of expected named parameter
                const auto valuep = namedArgs->Find(id);                    // was such parameter passed?
                const auto value = valuep ? *valuep : namedParam.second;    // if not given then fall back to default
                actualNamedArgs->Add(id, value.GetLocation(), value);
                // BUGBUG: we should pass in the location of the identifier, not that of the expression
            }
            for (const auto & namedArg : namedArgs->GetMembers())   // make sure there are no extra named args that the macro does not take
                if (namedParams->Find(namedArg.first) == nullptr)
                    throw EvaluationError(L"function does not have an optional argument '" + namedArg.first + L"'", namedArg.second.GetLocation());
            return f(args, actualNamedArgs, exprName);
        }
    };
    typedef shared_ptr<ConfigLambda> ConfigLambdaPtr;

    // -----------------------------------------------------------------------
    // functions exposed by this module
    // -----------------------------------------------------------------------

    // understand and execute from the syntactic expression tree
    ConfigValuePtr Evaluate(ExpressionPtr);     // evaluate the expression tree
    void Do(ExpressionPtr e);                   // evaluate e.do
    shared_ptr<Object> EvaluateField(ExpressionPtr e, const wstring & id);  // for experimental CNTK integration

}}}} // end namespaces
