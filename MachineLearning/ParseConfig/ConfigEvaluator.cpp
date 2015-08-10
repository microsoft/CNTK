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

    struct HasLateInit { virtual void Init(const ConfigRecord & config) = 0; }; // derive from this to indicate late initialization

    // dummy implementation of ComputationNode for experimental purposes
    struct Matrix { size_t rows; size_t cols; Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) { } };
    typedef shared_ptr<Matrix> MatrixPtr;

    struct ComputationNode : public Object, public HasToString
    {
        typedef shared_ptr<ComputationNode> ComputationNodePtr;

        // inputs and output
        vector<ComputationNodePtr> m_children;  // these are the inputs
        MatrixPtr m_functionValue;              // this is the result

        // other
        wstring m_nodeName;                     // node name in the graph

        virtual const wchar_t * TypeName() const = 0;

        virtual void AttachInputs(ComputationNodePtr leftNode, ComputationNodePtr rightNode)
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }

        /*implement*/ wstring ToString() const
        {
            return wstrprintf(L"%ls (%d inputs)", TypeName(), (int)m_children.size());
        }
    };
    typedef ComputationNode::ComputationNodePtr ComputationNodePtr;
    class BinaryComputationNode : public ComputationNode
    {
    public:
        BinaryComputationNode(ComputationNodePtr left, ComputationNodePtr right)
        {
            AttachInputs(left, right);
        }
    };
    class PlusNode : public BinaryComputationNode
    {
    public:
        PlusNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * TypeName() const { return L"PlusNode"; }
    };
    class MinusNode : public BinaryComputationNode
    {
    public:
        MinusNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * TypeName() const { return L"MinusNode"; }
    };
    class TimesNode : public BinaryComputationNode
    {
    public:
        TimesNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * TypeName() const { return L"TimesNode"; }
    };
#if 0   // ScaleNode is something more complex it seems
    class ScaleNode : public ComputationNode
    {
        double factor;
    public:
        TimesNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * TypeName() const { return L"ScaleNode"; }
    };
#endif
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
        /*implement*/ const wchar_t * TypeName() const { return L"DelayNode"; }
    };
    class InputValue : public ComputationNode
    {
    public:
        InputValue(const ConfigRecord & config)
        {
            config;
        }
        /*implement*/ const wchar_t * TypeName() const { return L"InputValue"; }
    };
    class LearnableParameter : public ComputationNode
    {
    public:
        LearnableParameter(size_t inDim, size_t outDim)
        {
            outDim; inDim;
        }
        /*implement*/ const wchar_t * TypeName() const { return L"LearnableParameter"; }
    };
    // factory function for ComputationNodes
    template<>
    shared_ptr<ComputationNode> MakeRuntimeObject<ComputationNode>(const ConfigRecord & config)
    {
        let classIdParam = config[L"class"];
        wstring classId = classIdParam;
        if (classId == L"LearnableParameter")
            return make_shared<LearnableParameter>(config[L"outDim"], config[L"inDim"]);
        else if (classId == L"PlusNode")
            return make_shared<PlusNode>((ComputationNodePtr)config[L"left"], (ComputationNodePtr)config[L"right"]);
        else if (classId == L"MinusNode")
            return make_shared<MinusNode>((ComputationNodePtr)config[L"left"], (ComputationNodePtr)config[L"right"]);
        else if (classId == L"TimesNode")
            return make_shared<TimesNode>((ComputationNodePtr)config[L"left"], (ComputationNodePtr)config[L"right"]);
#if 0
        else if (classId == L"ScaleNode")
            return make_shared<ScaleNode>((double)config[L"left"], (ComputationNodePtr)config[L"right"]);
#endif
        throw EvaluationError(L"unknown ComputationNode class " + classId, classIdParam.GetLocation());
    }

    // 'how' is the center of a printf format string, without % and type. Example %.2f -> how=".2"
    static wstring FormatConfigValue(ConfigValuePtr arg, const wstring & how)
    {
        size_t pos = how.find(L'%');
        if (pos != wstring::npos)
            RuntimeError("FormatConfigValue: format string must not contain %");
        if (arg.Is<String>())
        {
            return wstrprintf((L"%" + how + L"s").c_str(), arg.As<String>().c_str());
        }
        else if (arg.Is<Double>())
        {
            let val = arg.As<Double>();
            if (val == (int)val)
                return wstrprintf((L"%" + how + L"d").c_str(), (int)val);
            else
                return wstrprintf((L"%" + how + L"f").c_str(), val);
        }
        else if (arg.Is<ConfigArray>())
        {
            // TODO: this is not pretty at all
            let arr = arg.AsPtr<ConfigArray>();
            wstring result;
            let range = arr->GetRange();
            for (int i = range.first; i <= range.second; i++)
            {
                if (i > range.first)
                    result.append(L"\n");
                result.append(FormatConfigValue(arr->At(i, TextLocation()), how));
            }
            return result;
        }
        else if (arg.Is<HasToString>())
            return arg.As<HasToString>().ToString();
        else
            return msra::strfun::utf16(arg.TypeName());             // cannot print this type
    }

    // sample objects to implement functions
    class StringFunction : public String
    {
    public:
        StringFunction(const ConfigRecord & config)
        {
            wstring & us = *this;   // we write to this
            let arg = config[L"arg"];
            let whatArg = config[L"what"];
            wstring what = whatArg;
            if (what == L"format")
            {
                wstring how = config[L"how"];
                us = FormatConfigValue(arg, how);
            }
            else
                throw EvaluationError(L"unknown 'what' value to StringFunction: " + what, whatArg.GetLocation());
        }
    };

    // sample runtime objects for testing
    class PrintAction : public Object, public HasLateInit
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
            let what = config[L"what"];
            let str = what.Is<String>() ? what : FormatConfigValue(what, L""); // convert to string (without formatting information)
            fprintf(stderr, "%ls\n", str.c_str());
        }
    };

    class AnotherAction : public Object
    {
    public:
        AnotherAction(const ConfigRecord &) { fprintf(stderr, "Another\n"); }
        virtual ~AnotherAction(){}
    };

#if 0
    template<typename T> class BoxWithLateInitOf : public BoxOf<T>, public HasLateInit
    {
    public:
        BoxWithLateInitOf(T value) : BoxOf(value) { }
        /*implement*/ void Init(const ConfigRecord & config)
        {
            let hasLateInit = dynamic_cast<HasLateInit*>(BoxOf::value.get());
            if (!hasLateInit)
                LogicError("Init on class without HasLateInit");
            hasLateInit->Init(config);
        }
    };
#endif

    class Evaluator
    {
        // error handling

        __declspec(noreturn) void Fail(const wstring & msg, TextLocation where) { throw EvaluationError(msg, where); }

        __declspec(noreturn) void TypeExpected(const wstring & what, ExpressionPtr e) { Fail(L"expected expression of type " + what, e->location); }
        __declspec(noreturn) void UnknownIdentifier(const wstring & id, TextLocation where) { Fail(L"unknown member name " + id, where); }

        // lexical scope

        struct Scope
        {
            shared_ptr<ConfigRecord> symbols;   // symbols in this scope
            shared_ptr<Scope> up;               // one scope up
            Scope(shared_ptr<ConfigRecord> symbols, shared_ptr<Scope> up) : symbols(symbols), up(up) { }
        };
        typedef shared_ptr<Scope> ScopePtr;
        ScopePtr MakeScope(shared_ptr<ConfigRecord> symbols, shared_ptr<Scope> up) { return make_shared<Scope>(symbols, up); }

        // config value types

        // helper for configurableRuntimeTypes initializer below
        // This returns a lambda that is a constructor for a given runtime type.
        // LateInit currently broken.
        template<class C>
        function<ConfigValuePtr(const ConfigRecord &, TextLocation)> MakeRuntimeTypeConstructor()
        {
#if 0
            bool hasLateInit = is_base_of<HasLateInit, C>::value;   // (cannot test directly--C4127: conditional expression is constant)
            if (hasLateInit)
                return [this](const ConfigRecord & config, TextLocation location)
                {
                    return ConfigValuePtr(make_shared<BoxWithLateInitOf<shared_ptr<C>>>(make_shared<C>(config)), location);
                    return ConfigValuePtr(make_shared<C>(config), location);
            };
            else
#endif
                return [this](const ConfigRecord & config, TextLocation location)
                {
                    return ConfigValuePtr(MakeRuntimeObject<C>(config), location);
                };
        }

        // "new!" expressions get queued for execution after all other nodes of tree have been executed
        // TODO: This is totally broken, need to figuree out the deferred process first.
        struct LateInitItem
        {
            ConfigValuePtr object;
            ScopePtr scope;
            ExpressionPtr dictExpr;                             // the dictionary expression that now can be fully evaluated
            LateInitItem(ConfigValuePtr object, ScopePtr scope, ExpressionPtr dictExpr) : object(object), scope(scope), dictExpr(dictExpr) { }
        };

        // look up an identifier in an expression that is a ConfigRecord
        ConfigValuePtr RecordLookup(ExpressionPtr recordExpr, const wstring & id, TextLocation idLocation, ScopePtr scope)
        {
            let record = AsPtr<ConfigRecord>(Evaluate(recordExpr, scope), recordExpr, L"record");
            return ResolveIdentifier(id, idLocation, MakeScope(record, nullptr/*no up scope*/));
        }

        // evaluate all elements in a dictionary expression and turn that into a ConfigRecord
        // which is meant to be passed to the constructor or Init() function of a runtime object
        shared_ptr<ConfigRecord> ConfigRecordFromDictExpression(ExpressionPtr recordExpr, ScopePtr scope)
        {
            // evaluate the record expression itself
            // This will leave its members unevaluated since we do that on-demand
            // (order and what gets evaluated depends on what is used).
            let record = AsPtr<ConfigRecord>(Evaluate(recordExpr, scope), recordExpr, L"record");
            // resolve all entries, as they need to be passed to the C++ world which knows nothing about this
            record->ResolveAll();
            return record;
        }

        // perform late initialization
        // This assumes that the ConfigValuePtr points to a BoxWithLateInitOf. If not, it will fail with a nullptr exception.
        void LateInit(LateInitItem & lateInitItem)
        {
            let config = ConfigRecordFromDictExpression(lateInitItem.dictExpr, lateInitItem.scope);
            let object = lateInitItem.object;
            auto p = object.As<shared_ptr<HasLateInit>>();
            p->Init(*config);
//            dynamic_cast<HasLateInit*>(lateInitItem.object.get())->Init(*config);  // call BoxWithLateInitOf::Init() which in turn will call HasLateInite::Init() on the actual object
        }

        // get value
        // TODO: use &; does not currently work with AsBoxOfWrapped<ConfigRecord>
        template<typename T>
        T /*&*/ As(ConfigValuePtr value, ExpressionPtr e, const wchar_t * typeForMessage)
        {
            let val = dynamic_cast<T*>(value.get());
            if (!val)
                TypeExpected(typeForMessage, e);
            return *val;
        }
        // convert a BoxOfWrapped to a specific type
        // BUGBUG: If this returns a reference, it will crash when retrieving a ConfigRecord. May go away once ConfigRecord is used without Box
        template<typename T>
        T /*&*/ AsBoxOfWrapped(ConfigValuePtr value, ExpressionPtr e, const wchar_t * typeForMessage)
        {
            return As<BoxOfWrapped<T>>(value, e, typeForMessage);
            //let val = dynamic_cast<BoxOfWrapped<T>*>(value.get());
            //if (!val)
            //    TypeExpected(typeForMessage, e);
            //return *val;
        }
        template<typename T>
        shared_ptr<T> AsPtr(ConfigValuePtr value, ExpressionPtr e, const wchar_t * typeForMessage)
        {
            if (!value.Is<T>())
                TypeExpected(typeForMessage, e);
            return value.AsPtr<T>();
        }

        double ToDouble(ConfigValuePtr value, ExpressionPtr e) { return As<Double>(value, e, L"number"); }

        // get number and return it as an integer (fail if it is fractional)
        int ToInt(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = ToDouble(value, e);
            let res = (int)(val);
            if (val != res)
                TypeExpected(L"integer number", e);
            return res;
        }

#if 0
        // could just return String; e.g. same as To<String>
        wstring ToString(ConfigValuePtr value, ExpressionPtr e)
        {
            // TODO: shouldn't this be <String>?
            let val = dynamic_cast<String*>(value.get());
            if (!val)
                TypeExpected(L"string", e);
            return *val;
        }
#endif

        bool ToBoolean(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = dynamic_cast<Bool*>(value.get());            // TODO: factor out this expression
            if (!val)
                TypeExpected(L"boolean", e);
            return *val;
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

        // create a lambda that calls Evaluate() on an expr to get or realize its value
        ConfigValuePtr::Thunk MakeEvaluateThunk(ExpressionPtr expr, ScopePtr scope)
        {
            function<ConfigValuePtr()> f = [this, expr, scope]()   // lambda that computes this value of 'expr'
            {
                let value = Evaluate(expr, scope);
                return value;   // this is a great place to set a breakpoint!
            };
            return ConfigValuePtr::Thunk(f, expr->location);
        }

        // all infix operators with lambdas for evaluating them
        map<wstring, InfixFunctions> infixOps;

        // this table lists all C++ types that can be instantiated from "new" expressions
        map<wstring, function<ConfigValuePtr(const ConfigRecord &, TextLocation)>> configurableRuntimeTypes;

        // main evaluator function (highly recursive)
        //  - input:  expression
        //  - output: ConfigValuePtr that holds the evaluated value of the expression
        // Note that returned values may include complex value types like dictionaries (ConfigRecord) and functions (ConfigLambda).
        ConfigValuePtr Evaluate(ExpressionPtr e, ScopePtr scope)
        {
            // --- literals
            if (e->op == L"d")       return MakePrimitiveConfigValue(e->d, e->location);    // === double literal
            else if (e->op == L"s")  return MakeStringConfigValue(e->s, e->location);       // === string literal
            else if (e->op == L"b")  return MakePrimitiveConfigValue(e->b, e->location);    // === bool literal
            else if (e->op == L"new" || e->op == L"new!")                                   // === 'new' expression: instantiate C++ runtime object
            {
                // find the constructor lambda
                let newIter = configurableRuntimeTypes.find(e->id);
                if (newIter == configurableRuntimeTypes.end())
                    Fail(L"unknown runtime type " + e->id, e->location);
                // form the config record
                let dictExpr = e->args[0];
                if (e->op == L"new")   // evaluate the parameter dictionary into a config record
                    return newIter->second(*ConfigRecordFromDictExpression(dictExpr, scope), e->location); // this constructs it
                else                // ...unless it's late init. Then we defer initialization.
                {
                    // TODO: need a check here whether the class allows late init, before we actually try, so that we can give a concise error message
                    let value = newIter->second(ConfigRecord(), e->location);
                    deferredInitList.push_back(LateInitItem(value, scope, dictExpr)); // construct empty and remember to Init() later
                    return value;   // we return the created but not initialized object as the value, so others can reference it
                }
            }
            else if (e->op == L"if")                                                    // === conditional expression
            {
                let condition = ToBoolean(Evaluate(e->args[0], scope), e->args[0]);
                if (condition)
                    return Evaluate(e->args[1], scope);
                else
                    return Evaluate(e->args[2], scope);
            }
            // --- functions
            else if (e->op == L"=>")                                                    // === lambda (all macros are stored as lambdas)
            {
                // on scope: The lambda expression remembers the lexical scope of the '=>'; this is how it captures its context.
                let argListExpr = e->args[0];           // [0] = argument list ("()" expression of identifiers, possibly optional args)
                if (argListExpr->op != L"()") LogicError("parameter list expected");
                let fnExpr = e->args[1];                // [1] = expression of the function itself
                let f = [this, argListExpr, fnExpr, scope](const vector<ConfigValuePtr> & args, const shared_ptr<ConfigRecord> & namedArgs) -> ConfigValuePtr
                {
                    let & argList = argListExpr->args;
                    if (args.size() != argList.size()) LogicError("function application with mismatching number of arguments");
                    // create a ConfigRecord with param names from 'argList' and values from 'args'
                    // create a dictionary with all arguments
                    let record = make_shared<ConfigRecord>();
                    let thisScope = MakeScope(record, scope);   // look up in params first; then proceed upwards in lexical scope of '=>' (captured context)
                    // create an entry for every argument value
                    // Note that these values should normally be thunks since we only want to evaluate what's used.
                    for (size_t i = 0; i < args.size(); i++)    // positional arguments
                    {
                        let argName = argList[i];       // parameter name
                        if (argName->op != L"id") LogicError("function parameter list must consist of identifiers");
                        let & argVal = args[i];         // value of the parameter
                        record->Add(argName->id, argName->location, argVal);
                        // note: these are expressions for the parameter values; so they must be evaluated in the current scope
                    }
                    namedArgs;  // TODO: later
                    return Evaluate(fnExpr, MakeScope(record, scope));  // bring args into scope; keep lex scope of '=>' as upwards chain
                };
                let record = make_shared<ConfigRecord>();   // TODO: named args go here
                return ConfigValuePtr(make_shared<ConfigLambda>(argListExpr->args.size(), record, f), e->location);
            }
            else if (e->op == L"(")
            {
                let lambdaExpr = e->args[0];            // [0] = function
                let argsExpr = e->args[1];              // [1] = arguments passed to the function ("()" expression of expressions)
                let lambda = AsPtr<ConfigLambda>(Evaluate(lambdaExpr, scope), lambdaExpr, L"function");
                if (argsExpr->op != L"()") LogicError("argument list expected");
                // put all args into a vector of values
                // Like in an [] expression, we do not evaluate at this point, but pass in a lambda to compute on-demand.
                let args = argsExpr->args;
                if (args.size() != lambda->GetNumParams())
                    Fail(L"function parameter list must consist of identifiers", argsExpr->location);
                vector<ConfigValuePtr> argVals(args.size());
                for (size_t i = 0; i < args.size(); i++)    // positional arguments
                {
                    let argValExpr = args[i];               // expression of arg [i]
                    argVals[i] = MakeBoxedConfigValue(MakeEvaluateThunk(argValExpr, scope), argValExpr->location);  // make it a thunked value
                }
                // deal with namedArgs later
                let namedArgs = make_shared<ConfigRecord>();
#if 0
                for (let & entry : e->namedArgs)            // named args   --TODO: check whether arguments are matching and/or duplicate, use defaults
                    record->Add(entry.first, entry.second.first, MakeWrappedAndBoxedConfigValue(entry.second.second, entry.second.second->location));
                // BUGBUG: wrong text location passed in. Should be the one of the identifier, not the RHS. NamedArgs have no location.
#endif
                // call the function!
                return lambda->Apply(argVals, namedArgs);
            }
            // --- variable access
            else if (e->op == L"[]")                                                // === record (-> ConfigRecord)
            {
                let record = make_shared<ConfigRecord>();
                // create an entry for every dictionary entry.
                // We do not evaluate the members at this point.
                // Instead, as the value, we keep the ExpressionPtr itself.
                // Members are evaluated on demand when they are used.
                let thisScope = MakeScope(record, scope);       // lexical scope includes this dictionary itself, so we can access forward references
                for (let & entry : e->namedArgs)
                {
                    let expr = entry.second.second;                 // expression to compute the entry
                    record->Add(entry.first/*id*/, entry.second.first/*loc of id*/, MakeBoxedConfigValue(MakeEvaluateThunk(expr, thisScope), expr->location));
                }
                // BUGBUG: wrong text location passed in. Should be the one of the identifier, not the RHS. NamedArgs have no location.
                return ConfigValuePtr(record, e->location);
            }
            else if (e->op == L"id") return ResolveIdentifier(e->id, e->location, scope);   // === variable/macro access within current scope
            else if (e->op == L".")                                                 // === variable/macro access in given ConfigRecord element
            {
                let recordExpr = e->args[0];
                return RecordLookup(recordExpr, e->id, e->location, scope);
            }
            // --- arrays
            else if (e->op == L":")                                                 // === array expression (-> ConfigArray)
            {
                // this returns a flattened list of all members as a ConfigArray type
                let arr = make_shared<ConfigArray>();   // note: we could speed this up by keeping the left arg and appending to it
                for (let expr : e->args)        // concatenate the two args
                {
                    let item = Evaluate(expr, scope);           // result can be an item or a vector
                    if (item.Is<ConfigArray>())
                        arr->Append(item.As<ConfigArray>());     // append all elements (this flattens it)
                    else
                        arr->Append(item);
                }
                return ConfigValuePtr(arr, e->location);        // location will be that of the first ':', not sure if that is best way
            }
            else if (e->op == L"array")                                             // === array constructor from lambda function
            {
                let firstIndexExpr = e->args[0];    // first index
                let lastIndexExpr  = e->args[1];    // last index
                let initLambdaExpr = e->args[2];    // lambda to initialize the values
                let firstIndex = ToInt(Evaluate(firstIndexExpr, scope), firstIndexExpr);
                let lastIndex  = ToInt(Evaluate(lastIndexExpr, scope),  lastIndexExpr);
                let lambda = AsPtr<ConfigLambda>(Evaluate(initLambdaExpr, scope), initLambdaExpr, L"function");
                if (lambda->GetNumParams() != 1)
                    Fail(L"'array' requires an initializer function with one argument (the index)", initLambdaExpr->location);
                // At this point, we must know the dimensions and the initializer lambda, but we don't need to know all array elements.
                // Resolving array members on demand allows recursive access to the array variable, e.g. h[t] <- f(h[t-1]).
                // create a vector of Thunks to initialize each value
                vector<ConfigValuePtr> elementThunks;
                for (int index = firstIndex; index <= lastIndex; index++)
                {
                    let indexValue = MakePrimitiveConfigValue((double)index, e->location);      // index as a ConfigValuePtr
                    // create an expression
                    function<ConfigValuePtr()> f = [this, indexValue, initLambdaExpr, scope]()   // lambda that computes this value of 'expr'
                    {
                        // apply initLambdaExpr to indexValue and return the resulting value
                        let initLambda = AsPtr<ConfigLambda>(Evaluate(initLambdaExpr, scope), initLambdaExpr, L"function");
                        vector<ConfigValuePtr> argVals(1, indexValue);  // create an arg list with indexValue as the one arg
                        let namedArgs = make_shared<ConfigRecord>();    // no named args in initializer lambdas
                        let value = initLambda->Apply(argVals, namedArgs);
                        return value;   // this is a great place to set a breakpoint!
                    };
                    elementThunks.push_back(MakeBoxedConfigValue(ConfigValuePtr::Thunk(f, initLambdaExpr->location), initLambdaExpr->location));
                }
                auto arr = make_shared<ConfigArray>(firstIndex, move(elementThunks));
                return ConfigValuePtr(arr, e->location);
            }
            else if (e->op == L"[")                                                 // === access array element by index
            {
                let arrValue = Evaluate(e->args[0], scope);
                let indexExpr = e->args[1];
                let arr = AsPtr<ConfigArray>(arrValue, indexExpr, L"array");
                let dindex = As<Double>(Evaluate(indexExpr, scope), indexExpr, L"integer");
                let index = (int)dindex;
                if (index != dindex)
                    TypeExpected(L"integer", indexExpr);
                arr->ResolveValue(index, indexExpr->location);      // resolve each element only when it is used, to allow for recursive array access
                return arr->At(index, indexExpr->location);
            }
            // --- unary operators '+' '-' and '!'
            // ...
            // --- regular infix operators such as '+' and '=='
            else
            {
                let opIter = infixOps.find(e->op);
                if (opIter == infixOps.end())
                    LogicError("e->op " + utf8(e->op) + " not implemented");
                let & functions = opIter->second;
                let leftArg = e->args[0];
                let rightArg = e->args[1];
                let leftValPtr = Evaluate(leftArg, scope);
                let rightValPtr = Evaluate(rightArg, scope);
                if (leftValPtr.Is<Double>() && rightValPtr.Is<Double>())
                    return functions.NumbersOp(e, leftValPtr, rightValPtr);
                else if (leftValPtr.Is<String>() && rightValPtr.Is<String>())
                    return functions.StringsOp(e, leftValPtr, rightValPtr);
                else if (leftValPtr.Is<Bool>() && rightValPtr.Is<Bool>())
                    return functions.BoolOp(e, leftValPtr, rightValPtr);
                // ComputationNode is "magic" in that we map *, +, and - to know classes of fixed names.
                else if (leftValPtr.Is<ComputationNode>() && rightValPtr.Is<ComputationNode>())
                    return functions.ComputeNodeOp(e, leftValPtr, rightValPtr);
                else if (leftValPtr.Is<ComputationNode>() && rightValPtr.Is<Double>())
                    return functions.ComputeNodeNumberOp(e, leftValPtr, rightValPtr);
                else if (leftValPtr.Is<Double>() && rightValPtr.Is<ComputationNode>())
                    return functions.NumberComputeNodeOp(e, leftValPtr, rightValPtr);
                // TODO: DictOp
                else
                    FailBinaryOpTypes(e);
            }
            //LogicError("should not get here");
        }

        // look up a member by id in the search scope
        // If it is not found, it tries all lexically enclosing scopes inside out.
        const ConfigValuePtr & ResolveIdentifier(const wstring & id, TextLocation idLocation, ScopePtr scope)
        {
            if (!scope)                                         // no scope or went all the way up: not found
                UnknownIdentifier(id, idLocation);
            auto p = scope->symbols->Find(id);                  // look up the name
            if (!p)
                return ResolveIdentifier(id, idLocation, scope->up);    // not found: try next higher scope
            // found it: resolve the value lazily (the value will hold a Thunk to compute its value upon first use)
            p->ResolveValue();          // the entry will know
            // now the value is available
            return *p;
        }

        // evaluate a Boolean expression (all types)
        template<typename T>
        ConfigValuePtr CompOp(ExpressionPtr e, const T & left, const T & right)
        {
            if (e->op == L"==")      return MakePrimitiveConfigValue(left == right, e->location);
            else if (e->op == L"!=") return MakePrimitiveConfigValue(left != right, e->location);
            else if (e->op == L"<")  return MakePrimitiveConfigValue(left <  right, e->location);
            else if (e->op == L">")  return MakePrimitiveConfigValue(left >  right, e->location);
            else if (e->op == L"<=") return MakePrimitiveConfigValue(left <= right, e->location);
            else if (e->op == L">=") return MakePrimitiveConfigValue(left >= right, e->location);
            else LogicError("unexpected infix op");
        }
        // directly instantiate a ComputationNode for the magic operators * + and - that are automatically translated.
        ConfigValuePtr MakeMagicComputationNode(const wstring & classId, TextLocation location, const ConfigValuePtr & left, const ConfigValuePtr & right)
        {
            // find creation lambda
            let newIter = configurableRuntimeTypes.find(L"ComputationNode");
            if (newIter == configurableRuntimeTypes.end())
                LogicError("unknown magic runtime-object class");
            // form the ConfigRecord
            ConfigRecord config;
            config.Add(L"class", location, ConfigValuePtr(make_shared<String>(classId), location));
            config.Add(L"left",  left.GetLocation(),  left);
            config.Add(L"right", right.GetLocation(), right);
            // instantiate
            return newIter->second(config, location);
        }

        // Traverse through the expression (parse) tree to evaluate a value.
        deque<LateInitItem> deferredInitList;
    public:
        Evaluator()
        {
#define DefineRuntimeType(T) { L#T, MakeRuntimeTypeConstructor<T>() }
            // lookup table for "new" expression
            configurableRuntimeTypes = decltype(configurableRuntimeTypes)
            {
                // ComputationNodes
                DefineRuntimeType(ComputationNode),
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
                let left  = leftVal.As<Double>();
                let right = rightVal.As<Double>();
                if (e->op == L"+")       return MakePrimitiveConfigValue(left + right, e->location);
                else if (e->op == L"-")  return MakePrimitiveConfigValue(left - right, e->location);
                else if (e->op == L"*")  return MakePrimitiveConfigValue(left * right, e->location);
                else if (e->op == L"/")  return MakePrimitiveConfigValue(left / right, e->location);
                else if (e->op == L"%")  return MakePrimitiveConfigValue(fmod(left, right), e->location);
                else if (e->op == L"**") return MakePrimitiveConfigValue(pow(left, right), e->location);
                else return CompOp<double> (e, left, right);
            };
            InfixFunction StrOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                let left  = leftVal.As<String>();
                let right = rightVal.As<String>();
                if (e->op == L"+")  return MakeStringConfigValue(left + right, e->location);
                else return CompOp<wstring>(e, left, right);
            };
            InfixFunction BoolOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                let left  = leftVal.As<Bool>();
                let right = rightVal.As<Bool>();
                if (e->op == L"||")       return MakePrimitiveConfigValue(left || right, e->location);
                else if (e->op == L"&&")  return MakePrimitiveConfigValue(left && right, e->location);
                else if (e->op == L"^")   return MakePrimitiveConfigValue(left ^  right, e->location);
                else return CompOp<bool>(e, left, right);
            };
            InfixFunction NodeOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal) -> ConfigValuePtr
            {
                // TODO: test this
                if (rightVal.Is<Double>())     // ComputeNode * scalar
                    swap(leftVal, rightVal);        // -> scalar * ComputeNode
                if (leftVal.Is<Double>())      // scalar * ComputeNode
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

        // TODO: deferred list not working at all.
        //       Do() just calls into EvaluateParse directly.
        //       Need to move this list into Evaluate() directly and figure it out.
        ConfigValuePtr EvaluateParse(ExpressionPtr e)
        {
            auto result = Evaluate(e, nullptr/*top scope*/);
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
            RecordLookup(e, L"do", e->location, nullptr);  // we evaluate the member 'do'
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
