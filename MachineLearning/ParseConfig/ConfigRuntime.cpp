// ConfigRuntime.cpp -- execute what's given in a config file

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ConfigRuntime.h"
#include <deque>
#include <functional>
#include <cmath>

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

    class Evaluator
    {
        // error handling

        class EvaluationError : public ConfigError
        {
        public:
            EvaluationError(const wstring & msg, TextLocation where) : ConfigError(utf8(msg), where) { }
            /*implement*/ const char * kind() const { return "evaluating"; }
        };

        void Fail(const wstring & msg, TextLocation where) { throw EvaluationError(msg, where); }

        void TypeExpected(const wstring & what, ExpressionPtr e) { Fail(L"expected expression of type " + what, e->location); }
        void UnknownIdentifier(const wstring & id, TextLocation where) { Fail(L"unknown member name " + id, where); }

        // config value types

        template<typename T> ConfigValuePtr MakeConfigValue(const T & val) { return make_shared<ConfigValue<T>>(val); }

        // helper for configurableRuntimeTypes initializer below
        // This returns a lambda that is a constructor for a given runtime type.
        template<class C>
        function<ConfigValuePtr(const ConfigRecord &)> MakeRuntimeTypeConstructor()
        {
            bool hasLateInit = is_base_of<HasLateInit, C>::value;   // (cannot test directly--C4127: conditional expression is constant)
            if (hasLateInit)
                return [this](const ConfigRecord & config){ return make_shared<ConfigValueWithLateInit<shared_ptr<C>>>(make_shared<C>(config)); };
            else
                return [this](const ConfigRecord & config){ return MakeConfigValue(make_shared<C>(config)); };
        }

        // "new!" expressions get queued for execution after all other nodes of tree have been executed
        struct LateInitItem
        {
            ConfigValuePtr object;
            ExpressionPtr dictExpr;                             // the dictionary expression that now can be fully evaluated
            LateInitItem(ConfigValuePtr object, ExpressionPtr dictExpr) : object(object), dictExpr(dictExpr) { }
        };

        // evaluate all elements in a dictionary and turn that into a ConfigRecord
        // BUGBUG: This must be memorized. That's what variables are for!
        ConfigRecord ConfigRecordFromNamedArgs(ExpressionPtr e)
        {
            if (e->op != L"[]")
                TypeExpected(L"record", e);
            ConfigRecord config;
            for (let & namedArg : e->namedArgs)
            {
                let value = Evaluate(namedArg.second);
                config.Add(namedArg.first, value);
            }
            return config;
        }

        // perform late initialization
        // This assumes that the ConfigValuePtr points to a ConfigValueWithLateInit. If not, it will fail with a nullptr exception.
        void LateInit(LateInitItem & lateInitItem)
        {
            ConfigRecord config = ConfigRecordFromNamedArgs(lateInitItem.dictExpr);
            dynamic_cast<HasLateInit*>(lateInitItem.object.get())->Init(config);     // call ConfigValueWithLateInit::Init() which in turn will call HasLateInite::Init() on the actual object
        }

        double ToDouble(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = dynamic_cast<ConfigValue<double>*>(value.get());
            if (!val)
                TypeExpected(L"number", e);
            return val->value;
        }

        // get number and return it as an integer (fail if it is fractional)
        long long ToInt(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = ToDouble(value, e);
            let res = (long long)(val);
            if (val != res)
                TypeExpected(L"integer number", e);
            return res;
        }

        wstring ToString(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = dynamic_cast<ConfigValue<wstring>*>(value.get());
            if (!val)
                TypeExpected(L"number", e);
            return val->value;
        }

        bool ToBoolean(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = dynamic_cast<ConfigValue<bool>*>(value.get());            // TODO: factor out this expression
            if (!val)
                TypeExpected(L"boolean", e);
            return val->value;
        }

        // check if ConfigValuePtr is of a certain type
        template<typename T>
        bool Is(const ConfigValuePtr & value)
        {
            return dynamic_cast<ConfigValue<T>*>(value.get()) != nullptr;
        }

        // check if ConfigValuePtr is of a certain type
        template<typename T>
        const T & As(const ConfigValuePtr & value)
        {
            return dynamic_cast<ConfigValue<T>*>(value.get())->value;
        }

        typedef function<ConfigValuePtr(ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal)> InfixFunction;
        struct InfixFunctions
        {
            InfixFunction NumbersOp;            // number OP number -> number
            InfixFunction StringsOp;            // string OP string -> string
            InfixFunction BoolOp;               // bool OP bool -> bool
            InfixFunction ComputeNodeOp;        // ComputeNode OP ComputeNode -> ComputeNode
            InfixFunction NumberComputeNodeOp;  // number OP ComputeNode -> ComputeNode, e.g. 3 * M
            InfixFunction ComputeNodeNumberOp;  // ComputeNode OP Number -> ComputeNode, e.g. M * 3
            InfixFunction CompOp;               // ANY OP ANY -> bool
            InfixFunction DictOp;               // dict OP dict
            InfixFunctions(InfixFunction NumbersOp, InfixFunction StringsOp, InfixFunction BoolOp, InfixFunction ComputeNodeOp, InfixFunction NumberComputeNodeOp, InfixFunction ComputeNodeNumberOp, InfixFunction CompOp, InfixFunction DictOp)
                : NumbersOp(NumbersOp), StringsOp(StringsOp), BoolOp(BoolOp), ComputeNodeOp(ComputeNodeOp), NumberComputeNodeOp(NumberComputeNodeOp), ComputeNodeNumberOp(ComputeNodeNumberOp), CompOp(CompOp), DictOp(DictOp) { }
        };

        void FailBinaryOpTypes(ExpressionPtr e)
        {
            Fail(L"operator " + e->op + L" cannot be applied to these operands", e->location);
        }

        // all infix operators with lambdas for evaluating them
        map<wstring, InfixFunctions> infixOps;

        // this table lists all C++ types that can be instantiated from "new" expressions
        map<wstring, function<ConfigValuePtr(const ConfigRecord &)>> configurableRuntimeTypes;

        ConfigValuePtr Evaluate(ExpressionPtr e)
        {
            // this evaluates any evaluation node
            if (e->op == L"d")      return MakeConfigValue(e->d);
            else if (e->op == L"s") return MakeConfigValue(e->s);
            else if (e->op == L"b") return MakeConfigValue(e->b);
            else if (e->op == L"new" || e->op == L"new!")
            {
                // find the constructor lambda
                let newIter = configurableRuntimeTypes.find(e->id);
                if (newIter == configurableRuntimeTypes.end())
                    Fail(L"unknown runtime type " + e->id, e->location);
                // form the config record
                let dictExpr = e->args[0];
                if (e->op == L"new")   // evaluate the parameter dictionary into a config record
                    return newIter->second(ConfigRecordFromNamedArgs(dictExpr)); // this constructs it
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
                let condition = ToBoolean(Evaluate(e->args[0]), e->args[0]);
                if (condition)
                    return Evaluate(e->args[1]);
                else
                    return Evaluate(e->args[2]);
            }
            else
            {
                let opIter = infixOps.find(e->op);
                if (opIter == infixOps.end())
                    LogicError("e->op " + utf8(e->op) + " not implemented");
                let & functions = opIter->second;
                let leftArg = e->args[0];
                let rightArg = e->args[1];
                let leftValPtr = Evaluate(leftArg);
                let rightValPtr = Evaluate(rightArg);
                if (Is<double>(leftValPtr) && Is<double>(rightValPtr))
                    return functions.NumbersOp(e, leftValPtr, rightValPtr);
                else if (Is<wstring>(leftValPtr) && Is<wstring>(rightValPtr))
                    return functions.StringsOp(e, leftValPtr, rightValPtr);
                else if (Is<bool>(leftValPtr) && Is<bool>(rightValPtr))
                    return functions.BoolOp(e, leftValPtr, rightValPtr);
                // TODO: switch on the types
                else
                    FailBinaryOpTypes(e);
            }
            LogicError("should not get here");
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

        template<typename T>
        ConfigValuePtr CompOp(ExpressionPtr e, const T & left, const T & right)
        {
            if (e->op == L"==")      return MakeConfigValue(left == right);
            else if (e->op == L"!=") return MakeConfigValue(left != right);
            else if (e->op == L"<")  return MakeConfigValue(left <  right);
            else if (e->op == L">")  return MakeConfigValue(left >  right);
            else if (e->op == L"<=") return MakeConfigValue(left <= right);
            else if (e->op == L">=") return MakeConfigValue(left >= right);
            else LogicError("unexpected infix op");
        }

        // Traverse through the expression (parse) tree to evaluate a value.
        deque<LateInitItem> deferredInitList;
    public:
        Evaluator()
        {
            // lookup table for "new" expression
            configurableRuntimeTypes = decltype(configurableRuntimeTypes)
            {
                { L"PrintAction", MakeRuntimeTypeConstructor<PrintAction>() },
                { L"AnotherAction", MakeRuntimeTypeConstructor<AnotherAction>() }
            };
            // lookup table for infix operators
            // helper lambdas for evaluating infix operators
            InfixFunction NumOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                let left = As<double>(leftVal);
                let right = As<double>(rightVal);
                if (e->op == L"+")       return MakeConfigValue(left + right);
                else if (e->op == L"-")  return MakeConfigValue(left - right);
                else if (e->op == L"*")  return MakeConfigValue(left * right);
                else if (e->op == L"/")  return MakeConfigValue(left / right);
                else if (e->op == L"%")  return MakeConfigValue(fmod(left, right));
                else if (e->op == L"**") return MakeConfigValue(pow(left, right));
                else return CompOp<double> (e, left, right);
            };
            InfixFunction StrOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                let left = As<wstring>(leftVal);
                let right = As<wstring>(rightVal);
                if (e->op == L"+")  return MakeConfigValue(left + right);
                else return CompOp<wstring>(e, left, right);
            };
            InfixFunction BoolOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                let left = As<bool>(leftVal);
                let right = As<bool>(rightVal);
                if (e->op == L"||")       return MakeConfigValue(left || right);
                else if (e->op == L"&&")  return MakeConfigValue(left && right);
                else if (e->op == L"^")   return MakeConfigValue(left ^  right);
                else return CompOp<bool>(e, left, right);
            };
            InfixFunction BadOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr { FailBinaryOpTypes(e); return nullptr; };
            infixOps = decltype(infixOps)
            {
                // NumbersOp StringsOp BoolOp ComputeNodeOp NumberComputeNodeOp ComputeNodeNumberOp CompOp DictOp
                // CompOp does not work, fix this. Use a different mechanism.
                { L"*",  InfixFunctions(NumOp, BadOp, BadOp,  BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"/",  InfixFunctions(NumOp, BadOp, BadOp,  BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L".*", InfixFunctions(NumOp, BadOp, BadOp,  BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"**", InfixFunctions(NumOp, BadOp, BadOp,  BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"%",  InfixFunctions(NumOp, BadOp, BadOp,  BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"+",  InfixFunctions(NumOp, StrOp, BadOp,  BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"-",  InfixFunctions(NumOp, BadOp, BadOp,  BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"==", InfixFunctions(NumOp, StrOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"!=", InfixFunctions(NumOp, StrOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"<",  InfixFunctions(NumOp, StrOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L">",  InfixFunctions(NumOp, StrOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"<=", InfixFunctions(NumOp, StrOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L">=", InfixFunctions(NumOp, StrOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"&&", InfixFunctions(BadOp, BadOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"||", InfixFunctions(BadOp, BadOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) },
                { L"^",  InfixFunctions(BadOp, BadOp, BoolOp, BadOp, BadOp, BadOp, BadOp, BadOp) }
            };
        }

        ConfigValuePtr EvaluateParse(ExpressionPtr e)
        {
            auto result = Evaluate(e);
            // The deferredInitList contains unresolved Expressions due to "new!". This is specifically needed to support ComputeNodes
            // (or similar classes) that need circular references, while allowing to be initialized late (construct them empty first).
            while (!deferredInitList.empty())
            {
                LateInit(deferredInitList.front());
                deferredInitList.pop_front();
            }
            return result;
        }

        void Do(ExpressionPtr e)
        {
            let doValueExpr = LookupDictMember(e, e->location, L"do"); // expr to compute 'do' member
            EvaluateParse(doValueExpr);
        }
    };

    ConfigValuePtr Evaluate(ExpressionPtr e)
    {
        return Evaluator().EvaluateParse(e);
    }

    // top-level entry
    // A config sequence X=A;Y=B;do=(A,B) is really parsed as [X=A;Y=B].do. That's the tree we get. I.e. we try to compute the 'do' member.
    // TODO: This is not good--constructors should always be fast to run. Do() should run after late initializations.
    void Do(ExpressionPtr e)
    {
        Evaluator().Do(e);
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
        let parserTest = L"do = new PrintAction [ message = if 13 > 42 || 12 > 1 then 'Hello World' + \"!\" else 'Oops?']";
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
