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

    // TODO: refine Thunk handling
    // Thunks may only be resolved in-place at places that are supposed to hold ConfigValuePtrs that are evaluated on demand, such as
    //  - ConfigRecord
    //  - ConfigArrays
    //  - ConfigLambdas (default values of named arguments)
    // ConfigValuePtrs with Thunks may not be stored anywhere else, and are not assignable.
    // TODO: add two assignment/copy constructors:
    //  - true assignment/copy: runtime-fail if a Thunk
    //  - move assignment/copy: OK (then the few places that generate ConfigValuePtrs with Thunks must move them around as rvalue references with std::move())

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
        //       Basically, ConfigValuePtr are not copyable when in Thunked state.
        //       BUGBUG: This causes issues with macro parmaeters. They are copied (by value), but we cannot resolve when passing because Delay() will fail with circular reference.
        wstring expressionName;     // the name reflects the path to reach this expression in the (possibly dynamically macro-expanded) expression tree
    public:
        void operator=(ConfigValuePtr && other)
        {
            (shared_ptr<Object>&)*this = move(other);
            location = move(other.location);
            expressionName = move(other.expressionName);
        }
        void operator=(const ConfigValuePtr & other)
        {
            if (other.GetThunk())
                LogicError("ConfigValuePtr::operator=() on unresolved object; ConfigValuePtr is not assignable until resolved");
            (shared_ptr<Object>&)*this = other;
            location = other.location;
            expressionName = other.expressionName;
        }
        ConfigValuePtr(ConfigValuePtr && other) { *this = move(other); }
        ConfigValuePtr(const ConfigValuePtr & other) { *this = other; }
        //ConfigValuePtr(const ConfigValuePtr & other);
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
            // TODO: change all these ResolveValue() calls to CheckResolved()
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
        Thunk * GetThunk() const { return dynamic_cast<Thunk*>(get()); }    // get Thunk object or nullptr if already resolved
        const ConfigValuePtr & ResolveValue() const   // (this is const but mutates the value if it resolves)
        {
            // call this when a a member might be as-of-yet unresolved, to evaluate it on-demand
            // get() is a pointer to a Thunk in that case, that is, a function object that yields the value
            const auto thunkp = GetThunk();   // is it a Thunk?
            if (thunkp)                             // value is a Thunk: we need to resolve
            {
                const auto value = thunkp->ResolveValue();      // completely replace ourselves with the actual result. This also releases the Thunk object
                const_cast<ConfigValuePtr&>(*this) = value;
                ResolveValue();                     // allow it to return another Thunk...
            }
            return *this;                           // return ourselves so we can access a value as p_resolved = p->ResolveValue()
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
        ConfigRecordPtr parentScope;           // we look up the chain
        ConfigRecord() { }  // must give a scope
    public:
        ConfigRecord(ConfigRecordPtr parentScope) : parentScope(parentScope) { }

        // regular lookup: just use record[id]
        // Note that this function does not resolve Thunks. Instead, an unresolved value will come back as a Thunk.
        // TODO: Maybe this is the solution to the copying problem of ConfigValuePtrs:
        //  - we should resolve here! Hence, any ConfigValuePtr ever obtained from a ConfigRecord would be resolved
        //  - since ConfigRecords are the only place where multiple users may find a shared ConfigValuePtr, this would resolve it
        //  - if one value gets assigned to another (X=Y) and Y is unresolved, it would get resolved in its 'Y' location and only after that copied to X;
        //    that is OK, resolved ConfigValuePtrs can be copied
        //  - this way, ConfigValuePtrs with Thunks would never be passed around, except at the very place where a Thunk is created
        //    TODO: verify this, and maybe even add a custom assignment operator that prevents ConfigValuePtrs with Thunks to be assigned
        // TODO:
        //  - the LateInit problem could be solved by DelayNode accepting a lambda instead of a value, where that lambda would return the node;
        //    and DelayNode's initializer would keep that lambda, and only call it upon FinalizeInit().
        /*IsConfigRecord::*/ const ConfigValuePtr & operator()(const wstring & id, wstring message) const   // e.g. confRec(L"name", L"This specifies the object's internal name.")
        {
            const auto memberIter = members.find(id);
            if (memberIter != members.end())
                return memberIter->second.ResolveValue();   // resolve upon access
            if (parentScope)
                return (*parentScope)[id];                  // not found but have parent: look it up there
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
                if (parentScope)
                    return parentScope->Find(id);
                else
                    return nullptr;
            else
                return &memberIter->second.ResolveValue();
        }
        bool empty() const { return members.empty(); }      // late-init object constructors can test this
        // add a member
        void Add(const wstring & id, TextLocation idLocation, const ConfigValuePtr & value) { members[id] = value; idLocation; }
        void Add(const wstring & id, TextLocation idLocation, ConfigValuePtr && value) { members[id] = move(value); idLocation; } // use this for unresolved ConfigPtrs
        // TODO: ^^ idLocation is meant to hold the text location of the identifier
        // get members; used for optional argument lookup and logging
        const map<wstring, ConfigValuePtr> & GetMembers() const { return members; }
        // member resolution
        //void ResolveAll()   // resolve all members; do this before handing a ConfigRecord to C++ code
        //{
        //    for (auto & member : members)
        //        member.second.ResolveValue();
        //}
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
        // TODO: get rid of this function, only used in one place
        const ConfigValuePtr & GetElemRef(int index, TextLocation indexLocation) const
        {
            if (index < firstIndex || index >= firstIndex + values.size())
                throw EvaluationError(L"index out of bounds", indexLocation);
            return values[(size_t)(index - firstIndex)].ResolveValue(); // resolve upon access
        }
    public:
        ConfigArray() : firstIndex(0) { }
        ConfigArray(int firstIndex, vector<ConfigValuePtr> && values) : firstIndex(firstIndex), values(move(values)) { }
        pair<int, int> GetRange() const { return make_pair(firstIndex, firstIndex+(int)values.size()-1); }
        // building the array from expressions: append an element or an array
        void Append(ConfigValuePtr value) { values.push_back(value); }
        void Append(const ConfigArray & other) { values.insert(values.end(), other.values.begin(), other.values.end()); }
        // get element at index, including bounds check
        const ConfigValuePtr & At(int index, TextLocation indexLocation) const { return GetElemRef(index, indexLocation); }
    };
    typedef shared_ptr<ConfigArray> ConfigArrayPtr;

    // -----------------------------------------------------------------------
    // ConfigLambda -- a lambda
    // -----------------------------------------------------------------------

    class ConfigLambda : public Object
    {
    public:
        typedef map<wstring, ConfigValuePtr> NamedParams;   // TODO: maybe even not use a typedef, just use the type
    private:
        // the function itself is a C++ lambda
        function<ConfigValuePtr(const vector<ConfigValuePtr> &, const NamedParams &, const wstring & exprName)> f;
        // inputs. This defines the interface to the function. Very simple in our case though.
        vector<wstring> paramNames;             // #parameters and parameter names (names are used for naming expressions only)
        NamedParams namedParams;   // lists named parameters with their default values. Named parameters are optional and thus always must have a default.
        // TODO: are these defaults already resolved? Or Thunked and resolved upon first use?
        // TODO: Change namedParams to a shared_ptr<map<wstring,ConfigValuePtr>>
    public:
        template<typename F>
        ConfigLambda(vector<wstring> && paramNames, NamedParams && namedParams, const F & f) : paramNames(move(paramNames)), namedParams(move(namedParams)), f(f) { }
        size_t GetNumParams() const { return paramNames.size(); }
        const vector<wstring> & GetParamNames() const { return paramNames; }    // used for expression naming
        // what this function does is call f() held in this object with the given arguments except optional arguments are verified and fall back to their defaults if not given
        ConfigValuePtr Apply(vector<ConfigValuePtr> args, const NamedParams & namedArgs, const wstring & exprName)
        {
            NamedParams actualNamedArgs;
            // actualNamedArgs is a filtered version of namedArgs that contains all optional args listed in namedParams,
            // falling back to their default if not given in namedArgs.
            // On the other hand, any name in namedArgs that is not found in namedParams should be rejected.
            for (const auto & namedParam : namedParams)
            {
                const auto & id = namedParam.first;                         // id of expected named parameter
                const auto valuei = namedArgs.find(id);                    // was such parameter passed?
                const auto & value = valuei != namedArgs.end() ? valuei->second : namedParam.second.ResolveValue();    // if not given then fall back to default
                // BUGBUG: default may not have been resolved? -> first do namedParam.second->Resolve()? which would resolve in-place
                actualNamedArgs[id] = value;
                //actualNamedArgs->Add(id, value.GetLocation(), value);
                // BUGBUG: we should pass in the location of the identifier, not that of the expression
            }
            for (const auto & namedArg : namedArgs)   // make sure there are no extra named args that the macro does not take
                if (namedParams.find(namedArg.first) == namedParams.end())
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
