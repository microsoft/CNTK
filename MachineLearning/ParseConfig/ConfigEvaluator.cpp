// ConfigEvaluator.cpp -- execute what's given in a config file

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ConfigEvaluator.h"
#include <deque>
#include <functional>
#include <memory>
#include <cmath>

#ifndef let
#define let const auto
#endif

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;
    using namespace msra::strfun;

    struct HasLateInit : public Object { virtual void Init(const ConfigRecord & config) = 0; }; // derive from this to indicate late initialization

    // dummy implementation of ComputationNode for experimental purposes
    struct Matrix { size_t rows; size_t cols; Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) { } };
    typedef shared_ptr<Matrix> MatrixPtr;

    struct ComputationNode : public Object
    {
        typedef shared_ptr<ComputationNode> ComputationNodePtr;

        // inputs and output
        vector<MatrixPtr> children;     // these are the inputs
        MatrixPtr functionValue;        // this is the result

        // other
        wstring nodeName;               // node name in the graph
    };
    typedef ComputationNode::ComputationNodePtr ComputationNodePtr;
    class BinaryComputationNode : public ComputationNode
    {
    public:
        BinaryComputationNode(const ConfigRecord & config)
        {
            let left = (ComputationNodePtr) config[L"left"];
            let right = (ComputationNodePtr) config[L"right"];
            left; right;
        }
    };
    class TimesNode : public BinaryComputationNode
    {
    public:
        TimesNode(const ConfigRecord & config) : BinaryComputationNode(config) { }
    };
    class PlusNode : public BinaryComputationNode
    {
    public:
        PlusNode(const ConfigRecord & config) : BinaryComputationNode(config) { }
    };
    class MinusNode : public BinaryComputationNode
    {
    public:
        MinusNode(const ConfigRecord & config) : BinaryComputationNode(config) { }
    };
    class DelayNode : public ComputationNode, public HasLateInit
    {
    public:
        DelayNode(const ConfigRecord & config)
        {
            if (!config.empty())
                Init(config);
        }
        /*override*/ void Init(const ConfigRecord & config)
        {
            let in = (ComputationNodePtr)config[L"in"];
            in;
            // dim?
        }
    };
    class InputValue : public ComputationNode
    {
    public:
        InputValue(const ConfigRecord & config)
        {
            config;
        }
    };
    class LearnableParameter : public ComputationNode
    {
    public:
        LearnableParameter(const ConfigRecord & config)
        {
            let outDim = (size_t)config[L"outDim"];
            let inDim = (size_t)config[L"inDim"];
            outDim; inDim;
        }
    };

    // 'how' is the center of a printf format string, without % and type. Example %.2f -> how=".2"
    static wstring FormatConfigValue(ConfigValuePtr arg, const wstring & how)
    {
        size_t pos = how.find(L'%');
        if (pos != wstring::npos)
            RuntimeError("FormatConfigValue: format string must not contain %");
        if (arg.IsConfigValue<wstring>())
        {
            return wstrprintf((L"%" + how + L"s").c_str(), arg.AsConfigValue<wstring>());
        }
        else if (arg.IsConfigValue<double>())
        {
            return wstrprintf((L"%" + how + L"f").c_str(), arg.AsConfigValue<double>());
        }
        return L"?";
    }

    // sample objects to implement functions
    class StringFunction : public String
    {
    public:
        StringFunction(const ConfigRecord & config)
        {
            wstring & us = *this;   // we write to this
            let arg = config[L"arg"];
            wstring what = config[L"what"];
            if (what == L"format")
            {
                wstring how = config[L"how"];
                us = FormatConfigValue(arg, how);
                // TODO: implement this
            }
        }
    };

    // sample runtime objects for testing
    class PrintAction : public HasLateInit
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
            let & what = config[L"what"];
            if (what.IsConfigValue<wstring>())
                fprintf(stderr, "%ls\n", ((wstring)what).c_str());
            else if (what.IsConfigValue<double>())
            {
                let val = (double)what;
                if (val == (long long)val)
                    fprintf(stderr, "%d\n", (int)val);
                else
                    fprintf(stderr, "%f\n", val);
            }
            else if (what.IsConfigValue<bool>())
                fprintf(stderr, "%s\n", (bool)what ? "true" : "false");
            else
                fprintf(stderr, "(%s)\n", what.TypeName());
        }
    };

    class AnotherAction : public Object
    {
    public:
        AnotherAction(const ConfigRecord &) { fprintf(stderr, "Another\n"); }
        virtual ~AnotherAction(){}
    };

    template<typename T> class ConfigValueWithLateInit : public Wrapped<T>, public HasLateInit
    {
    public:
        ConfigValueWithLateInit(T value) : Wrapped(value) { }
        /*implement*/ void Init(const ConfigRecord & config)
        {
            let hasLateInit = dynamic_cast<HasLateInit*>(Wrapped::value.get());
            if (!hasLateInit)
                LogicError("Init on class without HasLateInit");
            hasLateInit->Init(config);
        }
    };

    class Evaluator
    {
        // error handling

        __declspec(noreturn) void Fail(const wstring & msg, TextLocation where) { throw EvaluationError(msg, where); }

        __declspec(noreturn) void TypeExpected(const wstring & what, ExpressionPtr e) { Fail(L"expected expression of type " + what, e->location); }
        __declspec(noreturn) void UnknownIdentifier(const wstring & id, TextLocation where) { Fail(L"unknown member name " + id, where); }

        // config value types

        // helper for configurableRuntimeTypes initializer below
        // This returns a lambda that is a constructor for a given runtime type.
        template<class C>
        function<ConfigValuePtr(const ConfigRecord &,TextLocation)> MakeRuntimeTypeConstructor()
        {
#if 0       // for now
            bool hasLateInit = is_base_of<HasLateInit, C>::value;   // (cannot test directly--C4127: conditional expression is constant)
            if (hasLateInit)
                return [this](const ConfigRecord & config, TextLocation location)
                {
                    return ConfigValuePtr(make_shared<ConfigValueWithLateInit<shared_ptr<C>>>(make_shared<C>(config)), location);
                };
            else
#endif
                return [this](const ConfigRecord & config, TextLocation location)
                {
                    return MakeConfigValue(make_shared<C>(config), location);
                };
        }
        template<>
        function<ConfigValuePtr(const ConfigRecord &, TextLocation)> MakeRuntimeTypeConstructor<StringFunction>()
        {
#if 0       // for now
            bool hasLateInit = is_base_of<HasLateInit, C>::value;   // (cannot test directly--C4127: conditional expression is constant)
            if (hasLateInit)
                return [this](const ConfigRecord & config, TextLocation location)
            {
                return ConfigValuePtr(make_shared<ConfigValueWithLateInit<shared_ptr<C>>>(make_shared<C>(config)), location);
            };
            else
#endif
                return [this](const ConfigRecord & config, TextLocation location)
            {
                const auto r = ConfigValuePtr(make_shared<StringFunction>(config), location);
                return r;
//                return MakeConfigValue(make_shared<StringFunction>(config), location);
            };
        }

        // "new!" expressions get queued for execution after all other nodes of tree have been executed
        struct LateInitItem
        {
            ConfigValuePtr object;
            ExpressionPtr dictExpr;                             // the dictionary expression that now can be fully evaluated
            LateInitItem(ConfigValuePtr object, ExpressionPtr dictExpr) : object(object), dictExpr(dictExpr) { }
        };

        // look up an identifier in a Wrapped<ConfigRecord>
        ConfigValuePtr RecordLookup(ExpressionPtr recordExpr, const wstring & id, TextLocation idLocation)
        {
            let record = AsConfigValue<ConfigRecordPtr>(Evaluate(recordExpr), recordExpr, L"record");
            // add it to the name-resolution scope
            scopes.push_back(record);
            // look up the name
            let & configMember = ResolveIdentifier(id, idLocation);
            // remove it again
            scopes.pop_back();
            //return (ConfigValuePtr)configMember;
            return configMember;
        }

        // evaluate all elements in a dictionary expression and turn that into a ConfigRecord
        // which is meant to be passed to the constructor or Init() function of a runtime object
        ConfigRecordPtr ConfigRecordFromDictExpression(ExpressionPtr recordExpr)
        {
            // evaluate the record expression itself
            // This will leave its members unevaluated since we do that on-demand
            // (order and what gets evaluated depends on what is used).
            let record = AsConfigValue<ConfigRecordPtr>(Evaluate(recordExpr), recordExpr, L"record");
            // add it to the name-resolution scope
            scopes.push_back(record);
            // resolve all entries
            record->ResolveAll([this](ExpressionPtr exprToResolve) { return Evaluate(exprToResolve); });
            // remove it again
            scopes.pop_back();
            return record;
        }

        // perform late initialization
        // This assumes that the ConfigValuePtr points to a ConfigValueWithLateInit. If not, it will fail with a nullptr exception.
        void LateInit(LateInitItem & lateInitItem)
        {
            let config = ConfigRecordFromDictExpression(lateInitItem.dictExpr);
            dynamic_cast<HasLateInit*>(lateInitItem.object.get())->Init(*config);  // call ConfigValueWithLateInit::Init() which in turn will call HasLateInite::Init() on the actual object
        }

        // convert a Wrapped to a specific type
        template<typename T>
        T AsConfigValue(ConfigValuePtr value, ExpressionPtr e, const wchar_t * typeForMessage)
        {
            let val = dynamic_cast<Wrapped<T>*>(value.get());
            if (!val)
                TypeExpected(typeForMessage, e);
            return *val;
        }

        double ToDouble(ConfigValuePtr value, ExpressionPtr e) { return AsConfigValue<double>(value, e, L"number"); }

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
            let val = dynamic_cast<Wrapped<wstring>*>(value.get());
            if (!val)
                TypeExpected(L"number", e);
            return *val;
        }

        bool ToBoolean(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = dynamic_cast<Wrapped<bool>*>(value.get());            // TODO: factor out this expression
            if (!val)
                TypeExpected(L"boolean", e);
            return *val;
        }

        // check if ConfigValuePtr is of a certain type
        template<typename T>
        bool IsConfigValue(const ConfigValuePtr & value)
        {
            return dynamic_cast<Wrapped<T>*>(value.get()) != nullptr;
        }

        // check if ConfigValuePtr is of a certain type
        template<typename T>
        const T & AsConfigValue(const ConfigValuePtr & value)
        {
            return *dynamic_cast<Wrapped<T>*>(value.get());
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
            InfixFunction DictOp;               // dict OP dict
            InfixFunctions(InfixFunction NumbersOp, InfixFunction StringsOp, InfixFunction BoolOp, InfixFunction ComputeNodeOp, InfixFunction NumberComputeNodeOp, InfixFunction ComputeNodeNumberOp, InfixFunction DictOp)
                : NumbersOp(NumbersOp), StringsOp(StringsOp), BoolOp(BoolOp), ComputeNodeOp(ComputeNodeOp), NumberComputeNodeOp(NumberComputeNodeOp), ComputeNodeNumberOp(ComputeNodeNumberOp), DictOp(DictOp) { }
        };

        __declspec(noreturn)
        void FailBinaryOpTypes(ExpressionPtr e)
        {
            Fail(L"operator " + e->op + L" cannot be applied to these operands", e->location);
        }

        // all infix operators with lambdas for evaluating them
        map<wstring, InfixFunctions> infixOps;

        // this table lists all C++ types that can be instantiated from "new" expressions
        map<wstring, function<ConfigValuePtr(const ConfigRecord &, TextLocation)>> configurableRuntimeTypes;

        ConfigValuePtr Evaluate(ExpressionPtr e)
        {
            // this evaluates any evaluation node
            if (e->op == L"d")       return MakeConfigValue(e->d, e->location);
            else if (e->op == L"s")  return MakeConfigValue(e->s, e->location);
            else if (e->op == L"b")  return MakeConfigValue(e->b, e->location);
            else if (e->op == L"id") return ResolveIdentifier(e->id, e->location);  // access a variable within current scope
            else if (e->op == L"new" || e->op == L"new!")
            {
                // find the constructor lambda
                let newIter = configurableRuntimeTypes.find(e->id);
                if (newIter == configurableRuntimeTypes.end())
                    Fail(L"unknown runtime type " + e->id, e->location);
                // form the config record
                let dictExpr = e->args[0];
                if (e->op == L"new")   // evaluate the parameter dictionary into a config record
                    return newIter->second(*ConfigRecordFromDictExpression(dictExpr), e->location); // this constructs it
                else                // ...unless it's late init. Then we defer initialization.
                {
                    // TODO: need a check here whether the class allows late init, before we actually try, so that we can give a concise error message
                    let value = newIter->second(ConfigRecord(), e->location);
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
            else if (e->op == L"[]")    // construct ConfigRecord
            {
                let record = make_shared<ConfigRecord>();
                // create an entry for every dictionary entry.
                // We do not evaluate the members at this point.
                // Instead, as the value, we keep the ExpressionPtr itself.
                // Members are evaluated on demand when they are used.
                for (let & entry : e->namedArgs)
                    record->Add(entry.first, entry.second.first, MakeConfigValue(entry.second.second, entry.second.second->location));
                // BUGBUG: wrong text location passed in. Should be the one of the identifier, not the RHS. NamedArgs have no location.
                return MakeConfigValue(record, e->location);
            }
            else if (e->op == L".")     // access ConfigRecord element
            {
                let recordExpr = e->args[0];
                let idExpr = e->args[1];
                if (idExpr->op != L"id")
                    LogicError("invalid field selector expression, must be 'id'");
                let id = idExpr->id;
                return RecordLookup(recordExpr, id, idExpr->location);
            }
            else if (e->op == L":")     // array expression
            {
                // TODO: test this
                // this returns a flattened list of all members as a ConfigArray type
                ConfigArray array;
                for (let expr : e->args)        // concatenate the two args
                {
                    let item = Evaluate(expr);  // result can be an item or a vector
                    if (IsConfigValue<ConfigArray>(item))
                    {
                        let items = AsConfigValue<ConfigArray>(item);
                        array.insert(array.end(), items.begin(), items.end());
                    }
                    else
                        array.push_back(item);
                }
                return MakeConfigValue(array, e->location); // location will be that of the first ':', not sure if that is best way
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
                if (IsConfigValue<double>(leftValPtr) && IsConfigValue<double>(rightValPtr))
                    return functions.NumbersOp(e, leftValPtr, rightValPtr);
                else if (IsConfigValue<wstring>(leftValPtr) && IsConfigValue<wstring>(rightValPtr))
                    return functions.StringsOp(e, leftValPtr, rightValPtr);
                else if (IsConfigValue<bool>(leftValPtr) && IsConfigValue<bool>(rightValPtr))
                    return functions.BoolOp(e, leftValPtr, rightValPtr);
                // ComputationNode is "magic" in that we map *, +, and - to know classes of fixed names.
                else if (IsConfigValue<shared_ptr<ComputationNode>>(leftValPtr) && IsConfigValue<shared_ptr<ComputationNode>>(rightValPtr))
                    return functions.ComputeNodeOp(e, leftValPtr, rightValPtr);
                else if (IsConfigValue<shared_ptr<ComputationNode>>(leftValPtr) && IsConfigValue<double>(rightValPtr))
                    return functions.ComputeNodeNumberOp(e, leftValPtr, rightValPtr);
                else if (IsConfigValue<double>(leftValPtr) && IsConfigValue<shared_ptr<ComputationNode>>(rightValPtr))
                    return functions.NumberComputeNodeOp(e, leftValPtr, rightValPtr);
                // TODO: DictOp
                else
                    FailBinaryOpTypes(e);
            }
            //LogicError("should not get here");
        }

        // look up a member by id in the search scope
        // If it is not found, it tries all lexically enclosing scopes inside out.
        const ConfigValuePtr & ResolveIdentifier(const wstring & id, TextLocation idLocation)
        {
            for (auto iter = scopes.rbegin(); iter != scopes.rend(); iter++/*goes backwards*/)
            {
                auto p = (*iter)->Find(id);     // look up the name
                if (p)
                {
                    // resolve the value lazily
                    // If it is not yet resolved then the value holds an ExpressionPtr.
                    p->ResolveValue([this](ExpressionPtr exprToResolve) { return Evaluate(exprToResolve); });
                    // now the value is available
                    return *p;                  // return ConfigValuePtr, like record[id], which one can now type-cast etc.
                }
                // if not found then try next outer scope
            }
            UnknownIdentifier(id, idLocation);
        }

        // evaluate a Boolean expression (all types)
        template<typename T>
        ConfigValuePtr CompOp(ExpressionPtr e, const T & left, const T & right)
        {
            if (e->op == L"==")      return MakeConfigValue(left == right, e->location);
            else if (e->op == L"!=") return MakeConfigValue(left != right, e->location);
            else if (e->op == L"<")  return MakeConfigValue(left <  right, e->location);
            else if (e->op == L">")  return MakeConfigValue(left >  right, e->location);
            else if (e->op == L"<=") return MakeConfigValue(left <= right, e->location);
            else if (e->op == L">=") return MakeConfigValue(left >= right, e->location);
            else LogicError("unexpected infix op");
        }
        // directly instantiate a ComputationNode for the magic operators * + and - that are automatically translated.
        ConfigValuePtr MakeMagicComputationNode(const wstring & classId, TextLocation location, const ConfigValuePtr & left, const ConfigValuePtr & right)
        {
            // find creation lambda
            let newIter = configurableRuntimeTypes.find(classId);
            if (newIter == configurableRuntimeTypes.end())
                LogicError("unknown magic runtime-object class");
            // form the ConfigRecord
            ConfigRecord config;
            config.Add(L"left",  left.location,  left);
            config.Add(L"right", right.location, right);
            // instantiate
            return newIter->second(config, location);
        }

        // Traverse through the expression (parse) tree to evaluate a value.
        deque<LateInitItem> deferredInitList;
        deque<ConfigRecordPtr> scopes;  // last entry is closest scope to be searched first
    public:
        Evaluator()
        {
#define DefineRuntimeType(T) { L#T, MakeRuntimeTypeConstructor<T>() }
            // lookup table for "new" expression
            configurableRuntimeTypes = decltype(configurableRuntimeTypes)
            {
                // ComputationNodes
                DefineRuntimeType(TimesNode),
                DefineRuntimeType(PlusNode),
                DefineRuntimeType(MinusNode),
                DefineRuntimeType(DelayNode),
                DefineRuntimeType(InputValue),
                DefineRuntimeType(LearnableParameter),
                // Functions
                DefineRuntimeType(StringFunction),
                // Actions
                DefineRuntimeType(PrintAction),
                DefineRuntimeType(AnotherAction),
            };
            // lookup table for infix operators
            // helper lambdas for evaluating infix operators
            InfixFunction NumOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                let left  = AsConfigValue<double>(leftVal);
                let right = AsConfigValue<double>(rightVal);
                if (e->op == L"+")       return MakeConfigValue(left + right, e->location);
                else if (e->op == L"-")  return MakeConfigValue(left - right, e->location);
                else if (e->op == L"*")  return MakeConfigValue(left * right, e->location);
                else if (e->op == L"/")  return MakeConfigValue(left / right, e->location);
                else if (e->op == L"%")  return MakeConfigValue(fmod(left, right), e->location);
                else if (e->op == L"**") return MakeConfigValue(pow(left, right), e->location);
                else return CompOp<double> (e, left, right);
            };
            InfixFunction StrOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                let left  = AsConfigValue<wstring>(leftVal);
                let right = AsConfigValue<wstring>(rightVal);
                if (e->op == L"+")  return MakeConfigValue(left + right, e->location);
                else return CompOp<wstring>(e, left, right);
            };
            InfixFunction BoolOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                let left  = AsConfigValue<bool>(leftVal);
                let right = AsConfigValue<bool>(rightVal);
                if (e->op == L"||")       return MakeConfigValue(left || right, e->location);
                else if (e->op == L"&&")  return MakeConfigValue(left && right, e->location);
                else if (e->op == L"^")   return MakeConfigValue(left ^  right, e->location);
                else return CompOp<bool>(e, left, right);
            };
            InfixFunction NodeOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                // TODO: test this
                if (IsConfigValue<double>(rightVal))    // ComputeNode * scalar
                    swap(leftVal, rightVal);            // -> scalar * ComputeNode
                if (IsConfigValue<double>(leftVal))     // scalar * ComputeNode
                {
                    if (e->op == L"*")  return MakeMagicComputationNode(L"ScaleNode", e->location, leftVal, rightVal);
                    else LogicError("unexpected infix op");
                }
                else                                // ComputeNode OP ComputeNode
                {
                    if (e->op == L"+")       return MakeMagicComputationNode(L"PlusNode",  e->location,  leftVal, rightVal);
                    else if (e->op == L"-")  return MakeMagicComputationNode(L"MinusNode", e->location, leftVal, rightVal);
                    else if (e->op == L"*")  return MakeMagicComputationNode(L"TimesNode", e->location, leftVal, rightVal);
                    else LogicError("unexpected infix op");
                }
            };
            InfixFunction BadOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr { FailBinaryOpTypes(e); };
            infixOps = decltype(infixOps)
            {
                // NumbersOp StringsOp BoolOp ComputeNodeOp DictOp
                { L"*",  InfixFunctions(NumOp, BadOp, BadOp,  NodeOp, NodeOp, NodeOp, BadOp) },
                { L"/",  InfixFunctions(NumOp, BadOp, BadOp,  BadOp,  BadOp,  BadOp,  BadOp) },
                { L".*", InfixFunctions(NumOp, BadOp, BadOp,  BadOp,  BadOp,  BadOp,  BadOp) },
                { L"**", InfixFunctions(NumOp, BadOp, BadOp,  BadOp,  BadOp,  BadOp,  BadOp) },
                { L"%",  InfixFunctions(NumOp, BadOp, BadOp,  BadOp,  BadOp,  BadOp,  BadOp) },
                { L"+",  InfixFunctions(NumOp, StrOp, BadOp,  NodeOp, BadOp,  BadOp,  BadOp) },
                { L"-",  InfixFunctions(NumOp, BadOp, BadOp,  NodeOp, BadOp,  BadOp,  BadOp) },
                { L"==", InfixFunctions(NumOp, StrOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) },
                { L"!=", InfixFunctions(NumOp, StrOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) },
                { L"<",  InfixFunctions(NumOp, StrOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) },
                { L">",  InfixFunctions(NumOp, StrOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) },
                { L"<=", InfixFunctions(NumOp, StrOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) },
                { L">=", InfixFunctions(NumOp, StrOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) },
                { L"&&", InfixFunctions(BadOp, BadOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) },
                { L"||", InfixFunctions(BadOp, BadOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) },
                { L"^",  InfixFunctions(BadOp, BadOp, BoolOp, BadOp,  BadOp,  BadOp,  BadOp) }
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
            RecordLookup(e, L"do", e->location);  // we evaluate the member 'do'
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
