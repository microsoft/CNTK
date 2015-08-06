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

    class EvaluationError : public ConfigError
    {
    public:
        EvaluationError(const wstring & msg, TextLocation where) : ConfigError(utf8(msg), where) { }
        /*implement*/ const char * kind() const { return "evaluating"; }
    };

    static void Fail(const wstring & msg, TextLocation where) { throw EvaluationError(msg, where); }

    static void TypeExpected(const wstring & what, ExpressionPtr e) { Fail(L"expected expression of type " + what, e->location); }
    static void UnknownIdentifier(const wstring & id, TextLocation where) { Fail(L"unknown member name " + id, where); }

    // ConfigValue variants
    //class ConfigValueLiteral : public ConfigValueBase { };

    template<typename T> class ConfigValueLiteral : public ConfigValueBase
    {
    public:
        /*const*/ T value;
        ConfigValueLiteral(T value) : value(value) { }
    };
    ConfigRecord::ConfigMember::operator wstring() const { return As<ConfigValueLiteral<wstring>>()->value; }

    template<class T> ConfigValueLiteral<shared_ptr<T>> MakeConfigValuePtr(const ConfigRecord & config)
    {
        return new ConfigValueLiteral<shared_ptr<T>>(make_shared(config));
    }

    map<wstring,function<ConfigValuePtr(const ConfigRecord &)>> configurableRuntimeTypes =
    {
        { L"PrintAction", [](const ConfigRecord & config){ return make_shared<ConfigValueLiteral<shared_ptr<PrintAction>>>(make_shared<PrintAction>(config)); } }
    };

    // "new!" expressions get queued for execution after all other nodes of tree have been executed
    struct LateInitItem
    {
        ConfigValuePtr object;  // the object to late-initialize
        ExpressionPtr dictExpr; // the dictionary expression that now can be fully evaluated
        LateInitItem(ConfigValuePtr object, ExpressionPtr dictExpr) : object(object), dictExpr(dictExpr) { }
        void Init(deque<LateInitItem> & workList);
    };

    static ConfigValuePtr Evaluate(ExpressionPtr e, deque<LateInitItem> & workList);

    // evaluate all elements in a dictionary and turn that into a ConfigRecord
    // BUGBUG: This must be memorized. That's what variables are for!
    ConfigRecord ConfigRecordFromNamedArgs(ExpressionPtr e, deque<LateInitItem> & workList)
    {
        if (e->op != L"[]")
            TypeExpected(L"record", e);
        ConfigRecord config;
        for (let & namedArg : e->namedArgs)
        {
            let value = Evaluate(namedArg.second, workList);
            config.Add(namedArg.first, value);
        }
        return config;
    }

    void LateInitItem::Init(deque<LateInitItem> & workList)
    {
        ConfigRecord config = ConfigRecordFromNamedArgs(dictExpr, workList);
        let configValuePtr = object.get();
        configValuePtr;
        // BUGBUG: This is broken. How do we get the type back?
        dynamic_cast<HasLateInit*>(object.get())->Init(config);
    }

    // evaluate the "new" operator. Also used in late init.
    static ConfigValuePtr EvaluateNew(const wstring & op, ExpressionPtr e, deque<LateInitItem> & workList)
    {
        // find the constructor lambda
        let newIter = configurableRuntimeTypes.find(e->id);
        if (newIter == configurableRuntimeTypes.end())
            Fail(L"unknown runtime type " + e->id, e->location);
        // form the config record
        let dictExpr = e->args[0];
        if (op == L"new")   // evaluate the parameter dictionary into a config record
            return newIter->second(ConfigRecordFromNamedArgs(dictExpr, workList)); // this constructs it
        else                // ...unless it's late init. Then we defer initialization.
        {
            // TODO: need a check here whether the class allows late init
            let value = newIter->second(ConfigRecord());
            workList.push_back(LateInitItem(value, dictExpr)); // construct empty and remember to Init() later
            return value;   // we return the created but not initialized object as the value, so others can reference it
        }
    }

    static ConfigValuePtr Evaluate(ExpressionPtr e, deque<LateInitItem> & workList)
    {
        // this evaluates any evaluation node
        if (e->op == L"d") { return make_shared<ConfigValueLiteral<double>>(e->d); }
        else if (e->op == L"s") { return make_shared<ConfigValueLiteral<wstring>>(e->s); }
        else if (e->op == L"b") { return make_shared<ConfigValueLiteral<bool>>(e->b); }
        else if (e->op == L"new" || e->op == L"new!") return EvaluateNew(e->op, e, workList);
        LogicError("unknown e->op");
    }

    // Traverse through the expression (parse) tree to evaluate a value.
    ConfigValuePtr Evaluate(ExpressionPtr e)
    {
        deque<LateInitItem> workList;
        auto result = Evaluate(e, workList);
        // The workList contains unresolved Expressions due to "new!". This is specifically needed to support ComputeNodes
        // (or similar classes) that need circular references, while allowing to be initialized late (construct them empty first).
        while (!workList.empty())
        {
            workList.front().Init(workList);
            workList.pop_front();
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
        let parserTest = L"do = new /*!*/ PrintAction [ message = 'Hello World']";
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
