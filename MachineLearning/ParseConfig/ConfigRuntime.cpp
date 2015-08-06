// ConfigRuntime.cpp -- execute what's given in a config file

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ConfigRuntime.h"
#include <deque>
#include <functional>

#ifndef let
#define let const auto
#endif

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;
    using namespace msra::strfun;

    struct HasLateInit { virtual void Init(const ConfigRecord & config) = 0; }; // derive from this to indicate late initialization

    // sample runtime objects for testing
    class PrintAction : public ConfigurableRuntimeObject, public HasLateInit
    {
    public:
        PrintAction(const ConfigRecord & config)
        {
            if (!config.empty())
                Init(config);
        }
        // example of late init (makes no real sense for PrintAction, of course)
        /*implement*/ void Init(const ConfigRecord & config)
        {
            wstring message = config[L"message"];
            fprintf(stderr, "%ls\n", message.c_str());
        }
    };

    class AnotherAction
    {
    public:
        AnotherAction(const ConfigRecord &) { fprintf(stderr, "Another\n"); }
        virtual ~AnotherAction(){}
    };

    // error handling

    class EvaluationError : public ConfigError
    {
    public:
        EvaluationError(const wstring & msg, TextLocation where) : ConfigError(utf8(msg), where) { }
        /*implement*/ const char * kind() const { return "evaluating"; }
    };

    static void Fail(const wstring & msg, TextLocation where) { throw EvaluationError(msg, where); }

    static void TypeExpected(const wstring & what, ExpressionPtr e) { Fail(L"expected expression of type " + what, e->location); }
    static void UnknownIdentifier(const wstring & id, TextLocation where) { Fail(L"unknown member name " + id, where); }

    // config value types

    template<typename T> class ConfigValueWithLateInit : public ConfigValue<T>, public HasLateInit
    {
    public:
        ConfigValueWithLateInit(T value) : ConfigValue(value) { }
        /*implement*/ void Init(const ConfigRecord & config)
        {
            let hasLateInit = dynamic_cast<HasLateInit*>(ConfigValue::value.get());
            if (!hasLateInit)
                LogicError("Init on class without HasLateInit");
            hasLateInit->Init(config);
        }
    };

    template<class T> ConfigValue<shared_ptr<T>> MakeConfigValuePtr(const ConfigRecord & config)
    {
        return new ConfigValue<shared_ptr<T>>(make_shared(config));
    }

    // helper for configurableRuntimeTypes initializer below
    // This returns a lambda that is a constructor for a given runtime type.
    template<class C>
    function<ConfigValuePtr(const ConfigRecord &)> MakeRuntimeTypeConstructor()
    {
        bool hasLateInit = is_base_of<HasLateInit, C>::value;   // (cannot test directly--C4127: conditional expression is constant)
        if (hasLateInit)
            return [](const ConfigRecord & config){ return make_shared<ConfigValueWithLateInit<shared_ptr<C>>>(make_shared<C>(config)); };
        else
            return [](const ConfigRecord & config){ return make_shared<ConfigValue<shared_ptr<C>>>(make_shared<C>(config)); };
    }

    // this table lists all C++ types that can be instantiated from "new" expressions
    map<wstring, function<ConfigValuePtr(const ConfigRecord &)>> configurableRuntimeTypes =
    {
        { L"PrintAction", MakeRuntimeTypeConstructor<PrintAction>() },
        { L"AnotherAction", MakeRuntimeTypeConstructor<AnotherAction>() }
    };

    // "new!" expressions get queued for execution after all other nodes of tree have been executed
    class LateInitItem
    {
        ConfigValuePtr object;
        ExpressionPtr dictExpr;                             // the dictionary expression that now can be fully evaluated
    public:
        LateInitItem(ConfigValuePtr object, ExpressionPtr dictExpr) : object(object), dictExpr(dictExpr) { }
        void Init(deque<LateInitItem> & deferredInitList);
    };

    static ConfigValuePtr Evaluate(ExpressionPtr e, deque<LateInitItem> & deferredInitList);

    // evaluate all elements in a dictionary and turn that into a ConfigRecord
    // BUGBUG: This must be memorized. That's what variables are for!
    ConfigRecord ConfigRecordFromNamedArgs(ExpressionPtr e, deque<LateInitItem> & deferredInitList)
    {
        if (e->op != L"[]")
            TypeExpected(L"record", e);
        ConfigRecord config;
        for (let & namedArg : e->namedArgs)
        {
            let value = Evaluate(namedArg.second, deferredInitList);
            config.Add(namedArg.first, value);
        }
        return config;
    }

    // perform late initialization
    // This assumes that the ConfigValuePtr points to a ConfigValueWithLateInit. If not, it will fail with a nullptr exception.
    void LateInitItem::Init(deque<LateInitItem> & deferredInitList)
    {
        ConfigRecord config = ConfigRecordFromNamedArgs(dictExpr, deferredInitList);
        dynamic_cast<HasLateInit*>(object.get())->Init(config);     // call ConfigValueWithLateInit::Init() which in turn will call HasLateInite::Init() on the actual object
    }

    static bool ToBoolean(ConfigValuePtr value, ExpressionPtr e)
    {
        let val = dynamic_cast<ConfigValue<bool>*>(value.get());
        if (!val)
            TypeExpected(L"boolean", e);
        return val->value;
    }

    static ConfigValuePtr Evaluate(ExpressionPtr e, deque<LateInitItem> & deferredInitList)
    {
        // this evaluates any evaluation node
        if (e->op == L"d")      return make_shared<ConfigValue<double>>(e->d);
        else if (e->op == L"s") return make_shared<ConfigValue<wstring>>(e->s);
        else if (e->op == L"b") return make_shared<ConfigValue<bool>>(e->b);
        else if (e->op == L"new" || e->op == L"new!")
        {
            // find the constructor lambda
            let newIter = configurableRuntimeTypes.find(e->id);
            if (newIter == configurableRuntimeTypes.end())
                Fail(L"unknown runtime type " + e->id, e->location);
            // form the config record
            let dictExpr = e->args[0];
            if (e->op == L"new")   // evaluate the parameter dictionary into a config record
                return newIter->second(ConfigRecordFromNamedArgs(dictExpr, deferredInitList)); // this constructs it
            else                // ...unless it's late init. Then we defer initialization.
            {
                // TODO: need a check here whether the class allows late init, before we actually try, so that we can give a concise error message
                let value = newIter->second(ConfigRecord());
                deferredInitList.push_back(LateInitItem(value, dictExpr)); // construct empty and remember to Init() later
                return value;   // we return the created but not initialized object as the value, so others can reference it
            }
        }
        else if (e->op == L"if")
        {
            let condition = ToBoolean(Evaluate(e->args[0], deferredInitList), e->args[0]);
            if (condition)
                return Evaluate(e->args[1], deferredInitList);
            else
                Evaluate(e->args[2], deferredInitList);
        }
        LogicError("unknown e->op");
    }

    // Traverse through the expression (parse) tree to evaluate a value.
    ConfigValuePtr Evaluate(ExpressionPtr e)
    {
        deque<LateInitItem> deferredInitList;
        auto result = Evaluate(e, deferredInitList);
        // The deferredInitList contains unresolved Expressions due to "new!". This is specifically needed to support ComputeNodes
        // (or similar classes) that need circular references, while allowing to be initialized late (construct them empty first).
        while (!deferredInitList.empty())
        {
            deferredInitList.front().Init(deferredInitList);
            deferredInitList.pop_front();
        }
        return result;
    }

    // look up a member by id in a dictionary expression
    // If it is not found, it tries all lexically enclosing scopes inside out.
    ExpressionPtr LookupDictMember(ExpressionPtr dict, TextLocation idLocation, const wstring & id)
    {
        if (!dict)  // we recursively go up; only when we reach the top do we fail
            UnknownIdentifier(id, idLocation);
        let idIter = dict->namedArgs.find(id);
        if (idIter == dict->namedArgs.end())
            return LookupDictMember(dict->parent, idLocation, id);  // not found: try parent
        return idIter->second;  // found it
    }

    // top-level entry
    // A config sequence X=A;Y=B;do=(A,B) is really parsed as [X=A;Y=B].do. That's the tree we get. I.e. we try to compute the 'do' member.
    // TODO: This is not good--constructors should always be fast to run. Do() should run after late initializations.
    void Do(ExpressionPtr e)
    {
        let doValueExpr = LookupDictMember(e, e->location, L"do"); // expr to compute 'do' member
        Evaluate(doValueExpr);
    }

}}}     // namespaces

#if 1   // use this for standalone development of the parser
using namespace Microsoft::MSR::CNTK;

// experimenting

// Thunk is a proxy with a type cast for accessing its value.

template<typename T> class ThunkOf : public Thunk
{
public:
    shared_ptr<T> p;
    T* operator->() const { return p.get(); }
    T& operator*() const { return *p.get(); }
};

int wmain(int /*argc*/, wchar_t* /*argv*/[])
{
    // there is record of parameters
    // user wants to get a parameter
    // double x = config->GetParam("name", 0.0);
    try
    {
        //let parserTest = L"a=1\na1_=13;b=2 // cmt\ndo = new PrintAction [message='hello'];do1=(print\n:train:eval) ; x = array[1..13] (i=>1+i*print.message==13*42) ; print = new PrintAction [ message = 'Hello World' ]";
        let parserTest = L"do = new ! PrintAction [ message = 'Hello World']";
        let expr = ParseConfigString(parserTest);
        expr->Dump();
        Do(expr);
        //ParseConfigFile(L"c:/me/test.txt")->Dump();
    }
    catch (const ConfigError & err)
    {
        err.PrintError();
    }
    return EXIT_SUCCESS;
}
#endif
