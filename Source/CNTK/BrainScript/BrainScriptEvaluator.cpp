// BrainScriptEvaluator.cpp -- execute what's given in a config file

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"

#include "ScriptableObjects.h"
#include "BrainScriptEvaluator.h"
#include "BrainScriptParser.h"

#include <deque>
#include <set>
#include <functional>
#include <memory>
#include <cmath>

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace BS {

using namespace std;
using namespace msra::strfun;
using namespace Microsoft::MSR::CNTK;
using namespace Microsoft::MSR::ScriptableObjects;

static bool trace = false; // set to true to enable to get debug output

// =======================================================================
// Evaluator -- class for evaluating a syntactic parse tree
// Evaluation converts a parse tree from ParseConfigDictFromString/File() into a graph of live C++ objects.
// =======================================================================

// -----------------------------------------------------------------------
// error handling
// -----------------------------------------------------------------------

// error object

class EvaluationException : public ConfigException
{
public:
    EvaluationException(const wstring &msg, TextLocation where)
        : ConfigException(msg, where)
    {
    }
    /*Configerror::*/ const wchar_t *kind() const override
    {
        return L"evaluating";
    }
};

__declspec_noreturn static inline void EvaluationError(const wstring &msg, TextLocation where)
{
    //Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
    throw EvaluationException(msg, where);
}

__declspec_noreturn static void Fail(const wstring &msg, TextLocation where)
{
    EvaluationError(msg, where);
}
__declspec_noreturn static void TypeExpected(const wstring &what, ExpressionPtr e)
{
    Fail(L"expected expression of type '" + what + L"'", e->location);
}
__declspec_noreturn static void UnknownIdentifier(const wstring &id, TextLocation where)
{
    Fail(L"unknown identifier '" + id + L"'", where);
}

// create a function that will fail with an error message at the given text location
// This is used to abstract awat knowledge of TextLocations from ConfigValuePtr (which could arise out of a different system, such as a Python wrapper).
function<void(const wstring &)> MakeFailFn(const TextLocation &textLocation)
{
    return [textLocation](const wstring &msg)
    {
        Fail(msg, textLocation);
    };
}

// -----------------------------------------------------------------------
// access to ConfigValuePtr content with error messages
// -----------------------------------------------------------------------

// get value
template <typename T>
static shared_ptr<T> AsPtr(ConfigValuePtr value, ExpressionPtr e, const wchar_t *typeForMessage)
{
    if (!value.Is<T>())
        TypeExpected(typeForMessage, e);
    return value.AsPtr<T>();
}

static double ToDouble(ConfigValuePtr value, ExpressionPtr e)
{
    let val = dynamic_cast<Double *>(value.get());
    if (!val)
        TypeExpected(L"number", e);
    double &dval = *val;
    return dval; // great place to set breakpoint
}

// get number and return it as an integer (fail if it is fractional)
static int ToInt(ConfigValuePtr value, ExpressionPtr e)
{
    let val = ToDouble(value, e);
    let res = (int) (val);
    if (val != res)
        TypeExpected(L"integer", e);
    return res;
}

static bool ToBoolean(ConfigValuePtr value, ExpressionPtr e)
{
    let val = dynamic_cast<Bool *>(value.get()); // TODO: factor out this expression
    if (!val)
        TypeExpected(L"boolean", e);
    return *val;
}

// -----------------------------------------------------------------------
// configurable runtime types ("new" expression)
// -----------------------------------------------------------------------

// internal types (such as string functions)
#define DefineRuntimeType(T)                    \
    {                                           \
        L## #T, MakeRuntimeTypeConstructor<T>() \
    }
template <class C>
static ConfigurableRuntimeType MakeRuntimeTypeConstructor()
{
    ConfigurableRuntimeType rtInfo;
    rtInfo.construct = [](const IConfigRecordPtr &config) // lambda to construct
    {
        return MakeRuntimeObject<C>(config);
    };
    rtInfo.isConfigRecord = is_base_of<IConfigRecord, C>::value;
    return rtInfo;
}

// get information about configurable runtime types
static const ConfigurableRuntimeType *FindRuntimeTypeInfo(const wstring &typeId)
{
    return ConfigurableRuntimeTypeRegister::Find(typeId);
}

// -----------------------------------------------------------------------
// name lookup
// -----------------------------------------------------------------------

static ConfigValuePtr Evaluate(const ExpressionPtr &e, const IConfigRecordPtr &scope, wstring exprPath, const wstring &exprId); // forward declare

// look up a member by id in the search scope
// If it is not found, it tries all lexically enclosing scopes inside out. This is handled by the ConfigRecord itself.
static const ConfigValuePtr &ResolveIdentifier(const wstring &id, const TextLocation &idLocation, const IConfigRecordPtr &scope)
{
    auto p = scope->Find(id); // look up the name
    // Note: We could also just use scope->operator[] here, like any C++ consumer, but then we'd not be able to print an error with a proper text location (that of the offending field).
    if (!p)
        UnknownIdentifier(id, idLocation);
    // found it: resolve the value lazily (the value will hold a Thunk to compute its value upon first use)
    p->EnsureIsResolved(); // if this is the first access, then the value must have executed its Thunk
    // now the value is available
    return *p;
}

// look up an identifier in an expression that is a ConfigRecord
static ConfigValuePtr RecordLookup(const ExpressionPtr &recordExpr, const wstring &id, const TextLocation &idLocation, const IConfigRecordPtr &scope, const wstring &exprPath)
{
    // Note on scope: The record itself (left of '.') must still be evaluated, and for that, we use the current scope;
    // that is, variables inside that expression--often a single variable referencing something in the current scope--
    // will be looked up there.
    // Now, the identifier on the other hand is looked up in the record and *its* scope (parent chain).
    let record = AsPtr<IConfigRecord>(Evaluate(recordExpr, scope, exprPath, L""), recordExpr, L"record");
    return ResolveIdentifier(id, idLocation, record /*resolve in scope of record; *not* the current scope*/);
}

// -----------------------------------------------------------------------
// runtime-object creation
// -----------------------------------------------------------------------

// evaluate all elements in a dictionary expression and turn that into a ConfigRecord
// which is meant to be passed to the constructor or Init() function of a runtime object
static shared_ptr<ConfigRecord> ConfigRecordFromDictExpression(const ExpressionPtr &recordExpr, const IConfigRecordPtr &scope, const wstring &exprPath)
{
    // evaluate the record expression itself
    // This will leave its members unevaluated since we do that on-demand
    // (order and what gets evaluated depends on what is used).
    let record = AsPtr<ConfigRecord>(Evaluate(recordExpr, scope, exprPath, L""), recordExpr, L"record");
    // resolve all entries, as they need to be passed to the C++ world which knows nothing about this
    return record;
}

// -----------------------------------------------------------------------
// infix operators
// -----------------------------------------------------------------------

// entry for infix-operator lookup table
typedef function<ConfigValuePtr(const ExpressionPtr &e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const IConfigRecordPtr &scope, const wstring &exprPath)> InfixOp /*const*/;
struct InfixOps
{
    wstring prettyName;    // pretty-printable name of this op, e.g. "Plus" for +
    InfixOp NumbersOp;     // number OP number -> number
    InfixOp StringsOp;     // string OP string -> string
    InfixOp BoolOp;        // bool OP bool -> bool
    InfixOp ComputeNodeOp; // one operand is ComputeNode -> ComputeNode
    InfixOp DictOp;        // dict OP dict
    InfixOps(const wchar_t *name, InfixOp NumbersOp, InfixOp StringsOp, InfixOp BoolOp, InfixOp ComputeNodeOp, InfixOp DictOp)
        : prettyName(name), NumbersOp(NumbersOp), StringsOp(StringsOp), BoolOp(BoolOp), ComputeNodeOp(ComputeNodeOp), DictOp(DictOp)
    {
    }
};

// functions that implement infix operations
__declspec_noreturn static void InvalidInfixOpTypes(ExpressionPtr e)
{
    Fail(L"operator " + e->op + L" cannot be applied to these operands", e->location);
}
template <typename T>
static ConfigValuePtr CompOp(const ExpressionPtr &e, const T &left, const T &right, const IConfigRecordPtr &, const wstring &exprPath)
{
    if      (e->op == L"==") return MakePrimitiveConfigValuePtr(left == right, MakeFailFn(e->location), exprPath);
    else if (e->op == L"!=") return MakePrimitiveConfigValuePtr(left != right, MakeFailFn(e->location), exprPath);
    else if (e->op == L"<")  return MakePrimitiveConfigValuePtr(left < right, MakeFailFn(e->location), exprPath);
    else if (e->op == L">")  return MakePrimitiveConfigValuePtr(left > right, MakeFailFn(e->location), exprPath);
    else if (e->op == L"<=") return MakePrimitiveConfigValuePtr(left <= right, MakeFailFn(e->location), exprPath);
    else if (e->op == L">=") return MakePrimitiveConfigValuePtr(left >= right, MakeFailFn(e->location), exprPath);
    else LogicError("unexpected infix op");
}
static ConfigValuePtr NumOp(const ExpressionPtr &e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const IConfigRecordPtr &scope, const wstring &exprPath)
{
    let left = leftVal.AsRef<Double>();
    let right = rightVal.AsRef<Double>();
    if      (e->op == L"+")  return MakePrimitiveConfigValuePtr(left + right, MakeFailFn(e->location), exprPath);
    else if (e->op == L"-")  return MakePrimitiveConfigValuePtr(left - right, MakeFailFn(e->location), exprPath);
    else if (e->op == L"*")  return MakePrimitiveConfigValuePtr(left * right, MakeFailFn(e->location), exprPath);
    else if (e->op == L"/")  return MakePrimitiveConfigValuePtr(left / right, MakeFailFn(e->location), exprPath);
    else if (e->op == L"%")  return MakePrimitiveConfigValuePtr(fmod(left, right), MakeFailFn(e->location), exprPath);
    else if (e->op == L"**") return MakePrimitiveConfigValuePtr(pow(left, right), MakeFailFn(e->location), exprPath);
    else return CompOp<double>(e, left, right, scope, exprPath);
};
static ConfigValuePtr StrOp(const ExpressionPtr &e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const IConfigRecordPtr &scope, const wstring &exprPath)
{
    let left = leftVal.AsRef<String>();
    let right = rightVal.AsRef<String>();
    if (e->op == L"+") return ConfigValuePtr(make_shared<String>(left + right), MakeFailFn(e->location), exprPath);
    else return CompOp<wstring>(e, left, right, scope, exprPath);
};
static ConfigValuePtr BoolOp(const ExpressionPtr &e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const IConfigRecordPtr &scope, const wstring &exprPath)
{
    let left = leftVal.AsRef<Bool>();
    //let right = rightVal.AsRef<Bool>();   // we do this inline, as to get the same short-circuit semantics as C++ (if rightVal is thunked, it will remain so unless required for this operation)
    if      (e->op == L"||") return MakePrimitiveConfigValuePtr(left || rightVal.AsRef<Bool>(), MakeFailFn(e->location), exprPath);
    else if (e->op == L"&&") return MakePrimitiveConfigValuePtr(left && rightVal.AsRef<Bool>(), MakeFailFn(e->location), exprPath);
    else if (e->op == L"^")  return MakePrimitiveConfigValuePtr(left ^ rightVal.AsRef<Bool>(), MakeFailFn(e->location), exprPath);
    else return CompOp<bool>(e, left, rightVal.AsRef<Bool>(), scope, exprPath);
};
// NodeOps handle the magic CNTK types, that is, infix operations between ComputeNode objects.
// TODO: we should have automagic up-casting of 'double' values to Constant() nodes, e.g. to allow to say "1 - P" where P is a node.
static ConfigValuePtr NodeOp(const ExpressionPtr &e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const IConfigRecordPtr &scope, const wstring &exprPath)
{
    // special cases/overloads:
    //  - unary minus -> NegateNode
    //  - product with a scalar
    // TODO: test these two (code was updated after originally tested)
    wstring operationName;
    if (e->op == L"-(")
    {
        if (rightVal.get())
            LogicError("unexpected infix op");
        operationName = L"Negate";
    }
    else if (e->op == L"*")
    {
        if (rightVal.Is<Double>())   // ComputeNode * scalar
            swap(leftVal, rightVal); // -> scalar * ComputeNode
        if (leftVal.Is<Double>())
            operationName = L"Scale"; // scalar * ComputeNode
        else
            operationName = L"Times"; // ComputeNode * ComputeNode (matrix produt)
    }
    else // ComputeNode OP ComputeNode
    {
        if (e->op == L"+")
            operationName = L"Plus";
        else if (e->op == L"-")
            operationName = L"Minus";
        else if (e->op == L".*")
            operationName = L"ElementTimes";
        else
            LogicError("unexpected infix op");
    }
    // directly instantiate a ComputationNode for the magic operators * + and - that are automatically translated.
    // find creation lambda
    let rtInfo = FindRuntimeTypeInfo(L"ComputationNode");
    if (!rtInfo)
        LogicError("unknown magic runtime-object class");
    // form the ConfigRecord for the ComputeNode that corresponds to the operation
    auto config = make_shared<ConfigRecord>(scope, MakeFailFn(e->location));
    // Note on scope: This config holds the arguments of the XXXNode runtime-object instantiations.
    // When they fetch their parameters, they should only look in this record, not in any parent scope (if they don't find what they are looking for, it's a bug in this routine here).
    // The values themselves are already in ConfigValuePtr form, so we won't need any scope lookups there either.
    config->Add(L"operation", MakeFailFn(e->location), ConfigValuePtr(make_shared<String>(operationName), MakeFailFn(e->location), exprPath));
    let leftFailFn = leftVal.GetFailFn(); // report any error for this Constant object as belonging to the scalar factor's expression
    vector<ConfigValuePtr> inputs;
    if (operationName == L"Scale")
    {
        // if we scale, the first operand is a Double, and we must convert that into a 1x1 Constant
        // TODO: apply this more generally to all operators
        auto constantConfig = make_shared<ConfigRecord>(config, MakeFailFn(e->location));
        constantConfig->Add(L"operation", leftFailFn, ConfigValuePtr(make_shared<String>(L"LearnableParameter"), leftFailFn, exprPath));
        let one = MakePrimitiveConfigValuePtr(1.0, leftFailFn, exprPath);
        constantConfig->Add(L"rows", leftFailFn, one);
        constantConfig->Add(L"cols", leftFailFn, one);
        //constantConfig->Add(L"shape", leftFailFn, one);  // BUGBUG: rows,cols is no longer right, we need a TensorShape here
        constantConfig->Add(L"value", leftFailFn, leftVal);
        constantConfig->Add(L"learningRateMultiplier", leftFailFn, MakePrimitiveConfigValuePtr(0.0f, leftFailFn, exprPath));
        let value = ConfigValuePtr(rtInfo->construct(constantConfig), leftFailFn, exprPath);
        let valueWithName = dynamic_cast<HasName *>(value.get());
        if (valueWithName)
            valueWithName->SetName(value.GetExpressionName());
        leftVal = value; // and that's our actual left value
    }
    inputs.push_back(leftVal);
    if (operationName != L"Negate") // Negate only has one input (rightVal is a nullptr)
        inputs.push_back(rightVal);
    config->Add(L"inputs", leftFailFn, ConfigValuePtr(make_shared<ConfigArray>(0, move(inputs)), leftFailFn, exprPath));
    config->Add(L"tag", leftFailFn, ConfigValuePtr(make_shared<String>(), leftFailFn, exprPath)); // infix nodes have no tag
    if (operationName == L"Times")
    {
        let one = MakePrimitiveConfigValuePtr(1.0, leftFailFn, exprPath);
        config->Add(L"outputRank", leftFailFn, one);
    }
    // instantiate the ComputationNode
    let value = ConfigValuePtr(rtInfo->construct(config), MakeFailFn(e->location), exprPath);
    let valueWithName = dynamic_cast<HasName *>(value.get());
    if (valueWithName)
        valueWithName->SetName(value.GetExpressionName());
    return value;
};
static ConfigValuePtr DictOp(const ExpressionPtr &e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const IConfigRecordPtr &scope, const wstring &exprPath)
{
    if (e->op != L"with")
        LogicError("unexpected infix op");
    let left = leftVal.AsPtr<ConfigRecord>();
    let right = rightVal.AsPtr<ConfigRecord>();
    left;
    right;
    scope;
    exprPath; // TODO: create a composite dictionary
    return leftVal;
};
static ConfigValuePtr BadOp(const ExpressionPtr &e, ConfigValuePtr, ConfigValuePtr, const IConfigRecordPtr &, const wstring &)
{
    InvalidInfixOpTypes(e);
};

// lookup table for infix operators
// This lists all infix operators with lambdas for evaluating them.
static map<wstring, InfixOps> infixOps =
{
    // symbol  PrettyName                NumbersOp StringsOp BoolOp  ComputeNodeOp DictOp
    { L"with", InfixOps(L"With",         NumOp,    BadOp,    BadOp,  NodeOp,       DictOp) },
    { L"*",    InfixOps(L"Times",        NumOp,    BadOp,    BadOp,  NodeOp,       BadOp) },
    { L"/",    InfixOps(L"Div",          NumOp,    BadOp,    BadOp,  BadOp,        BadOp) },
    { L".*",   InfixOps(L"ElementTimes", BadOp,    BadOp,    BadOp,  NodeOp,       BadOp) },
    { L"**",   InfixOps(L"Pow",          NumOp,    BadOp,    BadOp,  BadOp,        BadOp) },
    { L"%",    InfixOps(L"Mod",          NumOp,    BadOp,    BadOp,  BadOp,        BadOp) },
    { L"+",    InfixOps(L"Plus",         NumOp,    StrOp,    BadOp,  NodeOp,       BadOp) },
    { L"-",    InfixOps(L"Minus",        NumOp,    BadOp,    BadOp,  NodeOp,       BadOp) },
    { L"==",   InfixOps(L"Equal",        NumOp,    StrOp,    BoolOp, BadOp,        BadOp) },
    { L"!=",   InfixOps(L"NotEqual",     NumOp,    StrOp,    BoolOp, BadOp,        BadOp) },
    { L"<",    InfixOps(L"LT",           NumOp,    StrOp,    BoolOp, BadOp,        BadOp) },
    { L">",    InfixOps(L"GT",           NumOp,    StrOp,    BoolOp, BadOp,        BadOp) },
    { L"<=",   InfixOps(L"LE",           NumOp,    StrOp,    BoolOp, BadOp,        BadOp) },
    { L">=",   InfixOps(L"GT",           NumOp,    StrOp,    BoolOp, BadOp,        BadOp) },
    { L"&&",   InfixOps(L"And",          BadOp,    BadOp,    BoolOp, BadOp,        BadOp) },
    { L"||",   InfixOps(L"Or",           BadOp,    BadOp,    BoolOp, BadOp,        BadOp) },
    { L"^",    InfixOps(L"Xor",          BadOp,    BadOp,    BoolOp, BadOp,        BadOp) }
};

// -----------------------------------------------------------------------
// thunked (delayed) evaluation
// -----------------------------------------------------------------------

// create a lambda that calls Evaluate() on an expr to get or realize its value
// Unresolved ConfigValuePtrs (i.e. containing a Thunk) may only be moved, not copied.
static ConfigValuePtr MakeEvaluateThunkPtr(const ExpressionPtr &expr, const IConfigRecordPtr &scope, const wstring &exprPath, const wstring &exprId)
{
    function<ConfigValuePtr()> f = [expr, scope, exprPath, exprId]() // lambda that computes this value of 'expr'
    {
        if (trace)
            TextLocation::Trace(expr->location, msra::strfun::wstrprintf(L"thunk SP=0x%p", &exprPath).c_str(), expr->op.c_str(), (exprPath + L":" + exprId).c_str());
        let value = Evaluate(expr, scope, exprPath, exprId);
        return value; // this is a great place to set a breakpoint!
    };
    return ConfigValuePtr::MakeThunk(f, MakeFailFn(expr->location), exprPath);
}

// -----------------------------------------------------------------------
// main evaluator function (highly recursive)
// -----------------------------------------------------------------------

// Evaluate()
//  - input:  expression
//  - output: ConfigValuePtr that holds the evaluated value of the expression
//  - secondary inputs:
//     - scope: parent ConfigRecord to pass on to nested ConfigRecords we create, for recursive name lookup
//     - exprPath, exprId: for forming the expression path
// On expression paths:
//  - expression path encodes the path through the expression tree
//  - this is meant to be able to give ComputationNodes a name for later lookup that behaves the same as looking up an object directly
//  - not all nodes get their own path, in particular nodes with only one child, e.g. "-x", that would not be useful to address
// Note that returned values may include complex value types like dictionaries (ConfigRecord) and functions (ConfigLambda).
// TODO: This implementation takes a lot of stack space. Should break into many sub-functions.
static ConfigValuePtr Evaluate(const ExpressionPtr &e, const IConfigRecordPtr &scope, wstring exprPath, const wstring &exprId)
{
    try // catch clause for this will catch error, inject this tree node's TextLocation, and rethrow
    {
        // expression names
        // Merge exprPath and exprId into one unless one is empty
        if (!exprPath.empty() && !exprId.empty())
            exprPath.append(L".");
        exprPath.append(exprId);
        // tracing
        if (trace)
            TextLocation::Trace(e->location, msra::strfun::wstrprintf(L"eval SP=0x%p", &exprPath).c_str(), e->op.c_str(), exprPath.c_str());
        // --- literals
        if (e->op == L"d")
            return MakePrimitiveConfigValuePtr(e->d, MakeFailFn(e->location), exprPath); // === double literal
        else if (e->op == L"s")
            return ConfigValuePtr(make_shared<String>(e->s), MakeFailFn(e->location), exprPath); // === string literal
        else if (e->op == L"b")
            return MakePrimitiveConfigValuePtr(e->b, MakeFailFn(e->location), exprPath); // === bool literal
        else if (e->op == L"new")                                                        // === 'new' expression: instantiate C++ runtime object right here
        {
            // find the constructor lambda
            let rtInfo = FindRuntimeTypeInfo(e->id);
            if (!rtInfo)
                Fail(L"unknown runtime type " + e->id, e->location);
            // form the config record
            let &dictExpr = e->args[0];
            let argsExprPath = rtInfo->isConfigRecord ? L"" : exprPath;                                                                                      // reset expr-name path if object exposes a dictionary
            let value = ConfigValuePtr(rtInfo->construct(ConfigRecordFromDictExpression(dictExpr, scope, argsExprPath)), MakeFailFn(e->location), exprPath); // this constructs it
            // if object has a name, we set it
            let valueWithName = dynamic_cast<HasName *>(value.get());
            if (valueWithName)
                valueWithName->SetName(value.GetExpressionName());
            return value; // we return the created but not initialized object as the value, so others can reference it
        }
        else if (e->op == L"if") // === conditional expression
        {
            let condition = ToBoolean(Evaluate(e->args[0], scope, exprPath, L"if"), e->args[0]);
            if (condition)
                return Evaluate(e->args[1], scope, exprPath, L""); // pass exprName through 'if' since only of the two exists
            else
                return Evaluate(e->args[2], scope, exprPath, L"");
        }
        // --- functions
        else if (e->op == L"=>") // === lambda (all macros are stored as lambdas)
        {
            // on scope: The lambda expression remembers the lexical scope of the '=>'; this is how it captures its context.
            let &argListExpr = e->args[0]; // [0] = argument list ("()" expression of identifiers, possibly optional args)
            if (argListExpr->op != L"()")
                LogicError("parameter list expected");
            let &fnExpr = e->args[1]; // [1] = expression of the function itself
            let f = [argListExpr, fnExpr, scope, exprPath](vector<ConfigValuePtr> &&args, ConfigLambda::NamedParams &&namedArgs, const wstring &callerExprPath) -> ConfigValuePtr
            {
                // TODO: document namedArgs--does it have a parent scope? Or is it just a dictionary? Should we just use a shared_ptr<map,ConfigValuPtr>> instead for clarity?
                // on exprName
                //  - 'callerExprPath' is the name to which the result of the fn evaluation will be assigned
                //  - 'exprPath' (outside) is the name of the macro we are defining this lambda under
                let &argList = argListExpr->args;
                if (args.size() != argList.size())
                    LogicError("function application with mismatching number of arguments");
                // To execute a function body with passed arguments, we
                //  - create a new scope that contains all positional and named args
                //  - then evaluate the expression with that scope
                //  - parent scope for this is the scope of the function definition (captured context)
                //    Note that the 'scope' variable in here (we are in a lambda) is the scope of the '=>' expression, that is, the macro definition.
                // create a ConfigRecord with param names from 'argList' and values from 'args'
                let argScope = make_shared<ConfigRecord>(scope, MakeFailFn(argListExpr->location)); // look up in params first; then proceed upwards in lexical scope of '=>' (captured context)
                // Note: ^^ The failfn in the ConfigRecord will report unknown variables by pointing to the location of the argList expression.
                // However, as long as we run this lambda inside BrainScript, the access will check by itself and instead print the location of the variable.
                // create an entry for every argument value
                // Note that these values should normally be thunks since we only want to evaluate what's used.
                for (size_t i = 0; i < args.size(); i++) // positional arguments
                {
                    let argName = argList[i]; // parameter name
                    if (argName->op != L"id")
                        LogicError("function parameter list must consist of identifiers");
                    auto argVal = move(args[i]); // value of the parameter
                    let failfn = argVal.GetFailFn();
                    argScope->Add(argName->id, MakeFailFn(argName->location), move(argVal));
                    // note: these are expressions for the parameter values; so they must be evaluated in the current scope
                }
                // also named arguments
                for (auto &namedArg : namedArgs)
                {
                    let id = namedArg.first;
                    auto argVal = move(namedArg.second);
                    let failfn = argVal.GetFailFn();         // note: do before argVal gets destroyed in the upcoming move()
                    argScope->Add(id, failfn, move(argVal)); // TODO: is the failfn the right one?
                }
                // get the macro name for the exprPath
                wstring macroId = exprPath;
                let pos = macroId.find(L".");
                if (pos != wstring::npos)
                    macroId.erase(0, pos + 1);
                // now evaluate the function
                return Evaluate(fnExpr, argScope, callerExprPath, L"" /*L"[" + macroId + L"]"*/); // bring args into scope; keep lex scope of '=>' as upwards chain
            };
            // positional args
            vector<wstring> paramNames;
            let &argList = argListExpr->args;
            for (let &arg : argList)
            {
                if (arg->op != L"id")
                    LogicError("function parameter list must consist of identifiers");
                paramNames.push_back(arg->id);
            }
            // named args
            // The nammedArgs in the definition lists optional arguments with their default values
            ConfigLambda::NamedParams namedParams;
            for (let &namedArg : argListExpr->namedArgs)
            {
                let &id = namedArg.first;
                //let & location = namedArg.second.first;   // location of identifier
                let &expr = namedArg.second.second; // expression to evaluate to get default value
                namedParams[id] = move(MakeEvaluateThunkPtr(expr, scope /*evaluate default value in context of definition*/, exprPath /*TODO??*/, id));
                //namedParams->Add(id, location/*loc of id*/, ConfigValuePtr(MakeEvaluateThunkPtr(expr, scope/*evaluate default value in context of definition*/, exprPath, id), expr->location, exprPath/*TODO??*/));
                // the thunk is called if the default value is ever used
            }
            return ConfigValuePtr(make_shared<ConfigLambda>(move(paramNames), move(namedParams), f), MakeFailFn(e->location), exprPath);
        }
        else if (e->op == L"(") // === apply a function to its arguments
        {
            let &lambdaExpr = e->args[0]; // [0] = function
            let &argsExpr = e->args[1];   // [1] = arguments passed to the function ("()" expression of expressions)
            let lambda = AsPtr<ConfigLambda>(Evaluate(lambdaExpr, scope, exprPath, L"" /*macros are not visible in expression names*/), lambdaExpr, L"function");
            if (argsExpr->op != L"()")
                LogicError("argument list expected");
            // put all args into a vector of values
            // Like in an [] expression, we do not evaluate at this point, but pass in a lambda to compute on-demand.
            let &args = argsExpr->args;
            if (args.size() != lambda->GetNumParams())
                Fail(wstrprintf(L"function expects %d positional parameters, %d were provided", (int) lambda->GetNumParams(), (int) args.size()), argsExpr->location);
            vector<ConfigValuePtr> argVals(args.size());
            //bool onlyOneArg = args.size() == 1 && argsExpr->namedArgs.empty();
            for (size_t i = 0; i < args.size(); i++) // positional arguments
            {
                let argValExpr = args[i]; // expression to evaluate arg [i]
                let argName = lambda->GetParamNames()[i];
                argVals[i] = move(MakeEvaluateThunkPtr(argValExpr, scope, exprPath /*TODO??*/, /*onlyOneArg ? L"" :*/ argName));
                // Make it a thunked value and pass by rvalue ref since unresolved ConfigValuePtrs may not be copied.
                /*this wstrprintf should be gone, this is now the exprName*/
                // Note on scope: macro arguments form a scope (ConfigRecord), the expression for an arg does not have access to that scope.
                // E.g. F(A,B) is used as F(13,A) then that A must come from outside, it is not the function argument.
                // This is a little inconsistent with real records, e.g. [ A = 13 ; B = A ] where this A now does refer to this record.
                // However, it is still the expected behavior, because in a real record, the user sees all the other names, while when
                // passing args to a function, he does not; and also the parameter names can depend on the specific lambda being used.
            }
            // named args are put into a ConfigRecord
            // We could check whether the named ars are actually accepted by the lambda, but we leave that to Apply() so that the check also happens for lambda calls from CNTK C++ code.
            let &namedArgs = argsExpr->namedArgs;
            ConfigLambda::NamedParams namedArgVals;
            // TODO: no scope here? ^^ Where does the scope come in? Maybe not needed since all values are already resolved? Document this!
            for (let namedArg : namedArgs)
            {
                let id = namedArg.first;              // id of passed in named argument
                let location = namedArg.second.first; // location of expression
                let expr = namedArg.second.second;    // expression of named argument
                namedArgVals[id] = move(MakeEvaluateThunkPtr(expr, scope, exprPath /*TODO??*/, id));
                // the thunk is evaluated when/if the passed actual value is ever used the first time
                // This array owns the Thunk, and passes it by styd::move() to Apply, since it is not allowed to copy unresolved ConfigValuePtrs.
                // Note on scope: same as above.
                // E.g. when a function declared as F(A=0,B=0) is called as F(A=13,B=A), then A in B=A is not A=13, but anything from above.
                // For named args, it is far less clear whether users would expect this. We still do it for consistency with positional args, which are far more common.
            }
            // call the function!
            return lambda->Apply(move(argVals), move(namedArgVals), exprPath);
        }
        // --- variable access
        else if (e->op == L"[]") // === record (-> ConfigRecord)
        {
            let newScope = make_shared<ConfigRecord>(scope, MakeFailFn(e->location)); // new scope: inside this record, all symbols from above are also visible
            // ^^ The failfn here will be used if C++ code uses operator[] to retrieve a value. It will report the text location where the record was defined.
            // create an entry for every dictionary entry.
            // We do not evaluate the members at this point.
            // Instead, as the value, we keep the ExpressionPtr itself wrapped in a lambda that evaluates that ExpressionPtr to a ConfigValuePtr when called.
            // Members are evaluated on demand when they are used.
            for (let &entry : e->namedArgs)
            {
                let &id = entry.first;
                let &expr = entry.second.second; // expression to compute the entry
                newScope->Add(id, MakeFailFn(entry.second.first /*loc of id*/), MakeEvaluateThunkPtr(expr, newScope /*scope*/, exprPath /*TODO??*/, id));
                // Note on scope: record assignments are like a "let rec" in F#/OCAML. That is, all record members are visible to all
                // expressions that initialize the record members. E.g. [ A = 13 ; B = A ] assigns B as 13, not to a potentially outer A.
                // (To explicitly access an outer A, use the slightly ugly syntax ...A)
            }
            // BUGBUG: wrong text location passed in. Should be the one of the identifier, not the RHS. NamedArgs store no location for their identifier.
            return ConfigValuePtr(newScope, MakeFailFn(e->location), exprPath);
        }
        else if (e->op == L"id")
            return ResolveIdentifier(e->id, e->location, scope); // === variable/macro access within current scope
        else if (e->op == L".")                                  // === variable/macro access in given ConfigRecord element
        {
            let &recordExpr = e->args[0];
            return RecordLookup(recordExpr, e->id, e->location, scope /*for evaluating recordExpr*/, exprPath);
        }
        // --- arrays
        else if (e->op == L":") // === array expression (-> ConfigArray)
        {
            // this returns a flattened list of all members as a ConfigArray type
            let arr = make_shared<ConfigArray>();       // note: we could speed this up by keeping the left arg and appending to it
            for (size_t i = 0; i < e->args.size(); i++) // concatenate the two args
            {
                let &expr = e->args[i];
                arr->Append(move(MakeEvaluateThunkPtr(expr, scope, msra::strfun::wstrprintf(L"%ls[%d]", exprPath.c_str(), i), L"")));
            }
            return ConfigValuePtr(arr, MakeFailFn(e->location), exprPath); // location will be that of the first ':', not sure if that is best way
        }
        else if (e->op == L"array") // === array constructor from lambda function
        {
            let &firstIndexExpr = e->args[0]; // first index
            let &lastIndexExpr = e->args[1];  // last index
            let &initLambdaExpr = e->args[2]; // lambda to initialize the values
            let firstIndex = ToInt(Evaluate(firstIndexExpr, scope, exprPath, L"array_first"), firstIndexExpr);
            let lastIndex = ToInt(Evaluate(lastIndexExpr, scope, exprPath, L"array_last"), lastIndexExpr);
            let lambda = AsPtr<ConfigLambda>(Evaluate(initLambdaExpr, scope, exprPath, L"_initializer"), initLambdaExpr, L"function");
            if (lambda->GetNumParams() != 1)
                Fail(L"'array' requires an initializer function with one argument (the index)", initLambdaExpr->location);
            // At this point, we must know the dimensions and the initializer lambda, but we don't need to know all array elements.
            // Resolving array members on demand allows recursive access to the array variable, e.g. h[t] <- f(h[t-1]).
            // create a vector of Thunks to initialize each value
            vector<ConfigValuePtr> elementThunks;
            for (int index = firstIndex; index <= lastIndex; index++)
            {
                let indexValue = MakePrimitiveConfigValuePtr((double) index, MakeFailFn(e->location), exprPath /*never needed*/); // index as a ConfigValuePtr
                let elemExprPath = exprPath.empty() ? L"" : wstrprintf(L"%ls[%d]", exprPath.c_str(), index);                      // expression name shows index lookup
                let initExprPath = exprPath.empty() ? L"" : wstrprintf(L"_lambda");                                               // expression name shows initializer with arg
                // create a lambda that realizes this array element
                function<ConfigValuePtr()> f = [indexValue, initLambdaExpr, scope, elemExprPath, initExprPath]() // lambda that computes this value of 'expr'
                {
                    if (trace)
                        TextLocation::PrintIssue(vector<TextLocation>(1, initLambdaExpr->location), L"", wstrprintf(L"index %d", (int) indexValue).c_str(), L"executing array initializer thunk");
                    // apply initLambdaExpr to indexValue and return the resulting value
                    let initLambda = AsPtr<ConfigLambda>(Evaluate(initLambdaExpr, scope, initExprPath, L""), initLambdaExpr, L"function"); // get the function itself (most of the time just a simple name)
                    vector<ConfigValuePtr> argVals(1, indexValue);                                                                         // create an arg list with indexValue as the one arg
                    // TODO: where does the current scope come in? Aren't we looking up in namedArgs directly?
                    let value = initLambda->Apply(move(argVals), ConfigLambda::NamedParams(), elemExprPath);
                    // TODO: change this ^^ to the const & version of Apply() once it is there
                    return value; // this is a great place to set a breakpoint!
                };
                elementThunks.push_back(ConfigValuePtr::MakeThunk(f, MakeFailFn(initLambdaExpr->location), elemExprPath /*TODO??*/));
            }
            auto arr = make_shared<ConfigArray>(firstIndex, move(elementThunks));
            return ConfigValuePtr(arr, MakeFailFn(e->location), exprPath);
        }
        else if (e->op == L"[") // === access array element by index
        {
            let arrValue = Evaluate(e->args[0], scope, exprPath, L"_vector");
            let &indexExpr = e->args[1];
            let arr = AsPtr<ConfigArray>(arrValue, indexExpr, L"array");
            let index = ToInt(Evaluate(indexExpr, scope, exprPath, L"_index"), indexExpr);
            return arr->At(index, MakeFailFn(indexExpr->location)); // note: the array element may be as of now unresolved; this resolved it
        }
        // --- unary operators '+' '-' and '!'
        else if (e->op == L"+(" || e->op == L"-(") // === unary operators + and -
        {
            let &argExpr = e->args[0];
            let argValPtr = Evaluate(argExpr, scope, exprPath, e->op == L"+(" ? L"" : L"_negate");
            // note on exprPath: since - has only one argument, we do not include it in the expessionPath
            if (argValPtr.Is<Double>())
                if (e->op == L"+(")
                    return argValPtr;
                else
                    return MakePrimitiveConfigValuePtr(-(double) argValPtr, MakeFailFn(e->location), exprPath);
            else if (argValPtr.Is<ComputationNodeObject>()) // -ComputationNode becomes NegateNode(arg)
                if (e->op == L"+(")
                    return argValPtr;
                else
                    return NodeOp(e, argValPtr, ConfigValuePtr(), scope, exprPath);
            else
                Fail(L"operator '" + e->op.substr(0, 1) + L"' cannot be applied to this operand (which has type " + msra::strfun::utf16(argValPtr.TypeName()) + L")", e->location);
        }
        else if (e->op == L"!(") // === unary operator !
        {
            let arg = ToBoolean(Evaluate(e->args[0], scope, exprPath, L"_not"), e->args[0]);
            return MakePrimitiveConfigValuePtr(!arg, MakeFailFn(e->location), exprPath);
        }
        // --- regular infix operators such as '+' and '=='
        else
        {
            let opIter = infixOps.find(e->op);
            if (opIter == infixOps.end())
                LogicError("e->op '%ls' not implemented", e->op.c_str());
            let &functions = opIter->second;
            let &leftArg = e->args[0];
            let &rightArg = e->args[1];
            let leftValPtr = Evaluate(leftArg, scope, exprPath, functions.prettyName + L"Args[0]");
            let rightValPtr = Evaluate(rightArg, scope, exprPath, functions.prettyName + L"Args[1]");
            if (leftValPtr.Is<Double>() && rightValPtr.Is<Double>())
                return functions.NumbersOp(e, leftValPtr, rightValPtr, scope, exprPath);
            else if (leftValPtr.Is<String>() && rightValPtr.Is<String>())
                return functions.StringsOp(e, leftValPtr, rightValPtr, scope, exprPath);
            else if (leftValPtr.Is<Bool>() && rightValPtr.Is<Bool>())
                return functions.BoolOp(e, leftValPtr, rightValPtr, scope, exprPath);
            // ComputationNode is "magic" in that we map *, +, and - to know classes of fixed names.
            else if (leftValPtr.Is<ComputationNodeObject>() && rightValPtr.Is<ComputationNodeObject>())
                return functions.ComputeNodeOp(e, leftValPtr, rightValPtr, scope, exprPath);
            else if (leftValPtr.Is<ComputationNodeObject>() && rightValPtr.Is<Double>())
                return functions.ComputeNodeOp(e, leftValPtr, rightValPtr, scope, exprPath);
            else if (leftValPtr.Is<Double>() && rightValPtr.Is<ComputationNodeObject>())
                return functions.ComputeNodeOp(e, leftValPtr, rightValPtr, scope, exprPath);
            else if (leftValPtr.Is<ConfigRecord>() && rightValPtr.Is<ConfigRecord>())
                return functions.DictOp(e, leftValPtr, rightValPtr, scope, exprPath);
            else
                InvalidInfixOpTypes(e);
        }
    }
    catch (ConfigException &err)
    {
        // in case of an error, we keep track of all parent locations in the parse as well, to make it easier for the user to spot the error
        err.AddLocation(e->location);
        throw;
    }
}

static ConfigValuePtr EvaluateParse(ExpressionPtr e)
{
    return Evaluate(e, IConfigRecordPtr(nullptr) /*top scope*/, L"", L"$");
}

// -----------------------------------------------------------------------
// external entry points to the evaluator module
// -----------------------------------------------------------------------

// top-level entry
// A config sequence X=A;Y=B;do=(A,B) is really parsed as [X=A;Y=B].do. That's the tree we get. I.e. we try to compute the 'do' member.
void Do(ExpressionPtr e)
{
    RecordLookup(e, L"do", e->location, nullptr, L"$"); // we evaluate the member 'do'
}

shared_ptr<Object> EvaluateField(ExpressionPtr e, const wstring &id)
{
    return RecordLookup(e, id, e->location, nullptr /*scope for evaluating 'e'*/, L""); // we evaluate the member 'do'
}

ConfigValuePtr Evaluate(ExpressionPtr e)
{
    return EvaluateParse(e);
}

// =======================================================================
// built-in BrainScript functions and actions
// =======================================================================

// -----------------------------------------------------------------------
// built-in functions (implemented as Objects that are also their value)
// -----------------------------------------------------------------------

static wstring FormatConfigValue(ConfigValuePtr arg, const wstring &how);

// StringFunction implements
//  - Format
//  - Chr(c) -- gives a string of one character with Unicode value 'c'
//  - Replace(s,what,withwhat) -- replace all occurences of 'what' with 'withwhat'
//  - Substr(s,begin,num) -- get a substring
// TODO: RegexReplace()
class StringFunction : public String
{
    static wstring Substr(const wstring &s, int ibegin, int inum)
    {
        // negative index indexes from end; index may exceed
        let begin = min(ibegin < 0 ? s.size() + ibegin : ibegin, s.size());
        // 'num' is allowed to exceed
        let num = min(inum < 0 ? SIZE_MAX : inum, s.size() - begin);
        return s.substr(begin, num);
    }
    // TODO: RegexReplace!
public:
    StringFunction(const IConfigRecordPtr &configp)
    {
        let &config = *configp;
        wstring &us = *this; // we write to this
        let arg = config[L"arg"];
        let whatArg = config[L"what"];
        wstring what = whatArg;
        if (what == L"Format")
            us = FormatConfigValue(arg, config[L"how"]);
        else if (what == L"Chr")
            us = wstring(1, (wchar_t)(double) arg);
        else if (what == L"Substr")
            us = Substr(arg, config[L"pos"], config[L"chars"]);
        else if (what == L"Replace")
            us = msra::strfun::ReplaceAll<wstring>(arg, config[L"replacewhat"], config[L"withwhat"]);
        else
            whatArg.Fail(L"Unknown 'what' value to StringFunction: " + what);
    }
};

// FormatConfigValue() -- helper to print a config value to log
// 'how' is the center of a printf format string, without % and type. Example %.2f -> how=".2"
// TODO: change to taking a regular format string and a :: array of args that are checked. Support d,e,f,g,x,c,s (s also for ToString()).
// TODO: :: array. Check if that is the right operator for e.g. Haskell.
// TODO: turn Print into PrintF; e.g. PrintF provides 'format' arg. Printf('solution to %s is %d', 'question' :: 42)
static wstring FormatConfigValue(ConfigValuePtr arg, const wstring &how)
{
    size_t pos = how.find(L'%');
    if (pos != wstring::npos)
        RuntimeError("FormatConfigValue: format string must not contain %%");
    if (arg.Is<String>())
    {
        return wstrprintf((L"%" + how + L"s").c_str(), arg.AsRef<String>().c_str());
    }
    else if (arg.Is<Double>())
    {
        let val = arg.AsRef<Double>();
        if (val == (int) val)
            return wstrprintf((L"%" + how + L"d").c_str(), (int) val);
        else
            return wstrprintf((L"%" + how + L"f").c_str(), val);
    }
    else if (arg.Is<ConfigRecord>()) // TODO: should have its own ToString() method
    {
        let record = arg.AsPtr<ConfigRecord>();
        let memberIds = record->GetMemberIds(); // TODO: test this after change to ids
        wstring result;
        bool first = true;
        for (let &id : memberIds)
        {
            if (first)
                first = false;
            else
                result.append(L"\n");
            result.append(id);
            result.append(L" = ");
            result.append(FormatConfigValue((*record)[id], how));
        }
        return HasToString::NestString(result, L'[', true, L']');
    }
    else if (arg.Is<ConfigArray>()) // TODO: should have its own ToString() method
    {
        let arr = arg.AsPtr<ConfigArray>();
        wstring result;
        let range = arr->GetIndexRange();
        for (int i = range.first; i <= range.second; i++)
        {
            if (i > range.first)
                result.append(L"\n");
            result.append(FormatConfigValue(arr->At(i, [](const wstring &) { LogicError("FormatConfigValue: out of bounds index while iterating??"); }), how));
        }
        return HasToString::NestString(result, L'(', false, L')');
    }
    else if (arg.Is<HasToString>())
        return arg.AsRef<HasToString>().ToString();
    else
        return msra::strfun::utf16(arg.TypeName()); // cannot print this type
}

// NumericFunctions
//  - Floor()
//  - Length() (of string or array)
class NumericFunction : public BoxOf<Double>
{
public:
    NumericFunction(const IConfigRecordPtr &configp) :
        BoxOf<Double>(0.0)
    {
        let &config = *configp;
        double &us = *this; // we write to this
        let arg = config[L"arg"];
        let whatArg = config[L"what"];
        wstring what = whatArg;
        if (what == L"Floor")
            us = floor((double) arg);
        else if (what == L"Length")
        {
            if (arg.Is<String>())
                us = (double) ((wstring &) arg).size();
            else // otherwise expect an array
            {
                let & arr = arg.AsRef<ConfigArray>();
                let range = arr.GetIndexRange();
                us = (double) (range.second + 1 - range.first);
            }
        }
        else
            whatArg.Fail(L"Unknown 'what' value to NumericFunction: " + what);
    }
};

// CompareFunctions
//  - IsSameObject()
class CompareFunction : public BoxOf<Bool>
{
public:
    CompareFunction(const IConfigRecordPtr &configp) :
        BoxOf<Bool>(false)
    {
        let &config = *configp;
        bool &us = *this; // we write to this
        let argsArg = config[L"args"];
        let whatArg = config[L"what"];
        wstring what = whatArg;
        if (what == L"IsSameObject")
        {
            let& args = argsArg.AsRef<ConfigArray>();
            auto range = args.GetIndexRange();
            if (range.second != range.first+1)
                argsArg.Fail(L"IsSameObject expects two arguments");
            let arg1 = args.At(range.first ).AsPtr<Object>();
            let arg2 = args.At(range.second).AsPtr<Object>();
            us = arg1.get() == arg2.get();
        }
        else
            whatArg.Fail(L"Unknown 'what' value to CompareFunction: " + what);
    }
};

// -----------------------------------------------------------------------
// general-purpose use Actions
// -----------------------------------------------------------------------

// sample runtime objects for testing
class PrintAction : public Object
{
public:
    PrintAction(const IConfigRecordPtr &configp)
    {
        let &config = *configp;
        let what = config[L"what"];
        let str = what.Is<String>() ? what : FormatConfigValue(what, L""); // convert to string (without formatting information)
        fprintf(stderr, "%ls\n", str.c_str());
    }
};

// FailAction just throw a config error
class FailAction : public Object
{
public:
    FailAction(const IConfigRecordPtr &configp)
    {
        let &config = *configp;
        // note: not quite optimal yet in terms of how the error is shown; e.g. ^ not showing under offending variable
        let messageValue = config[L"what"];
        bool fail = true;
        if (fail)                            // this will trick the VS compiler into not issuing warning 4702: unreachable code
            messageValue.Fail(messageValue); // this will show the location of the message string, which is next to the Fail() call
    }
};

// Debug is a special class that just dumps its argument's value to log and then returns that value
struct Debug : public Object
{
    Debug(const IConfigRecordPtr)
    {
    }
}; // fake class type to get the template below trigger
template <>
/*static*/ ConfigurableRuntimeType MakeRuntimeTypeConstructor<Debug>()
{
    ConfigurableRuntimeType rtInfo;
    rtInfo.construct = [](const IConfigRecordPtr &configp)
    {
        let &config = *configp;
        let value = config[L"value"];
        bool enabled = config[L"enabled"];
        if (enabled)
        {
            wstring say = config[L"say"];
            if (!say.empty())
                fprintf(stderr, "%ls\n", say.c_str());
            let str = value.Is<String>() ? value : FormatConfigValue(value, L""); // convert to string (without formatting information)
            fprintf(stderr, "%ls\n", str.c_str());
        }
        return value;
    };
    rtInfo.isConfigRecord = false;
    return rtInfo;
}

// =======================================================================
// register ComputationNetwork with the ScriptableObject system
// =======================================================================

// Functions
static ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<StringFunction>  registerStringFunction(L"StringFunction");
static ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<NumericFunction> registerNumericFunction(L"NumericFunction");
static ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<CompareFunction> registerCompareFunction(L"CompareFunction");
// Actions
static ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<PrintAction>     registerPrintAction(L"PrintAction");
static ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<FailAction>      registerFailAction(L"FailAction");
// Special
static ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<Debug>           registerDebug(L"Debug");

// main TODO items:
//  - break Evaluate() to optimize stack usage
//  - dictionary merging, to allow overwriting from command line
//     - [ d1 ] + [ d2 ] will install a filter in d1 to first check against d2
//     - d2 can have fully qualified names on the LHS, and the filter is part of a chain that is passed down to inner dictionaries created
//     - d1 + d2 == wrapper around d1 with filter(d2)
//       When processing [ ] expressions inside d1, the current filter chain is applied straight away.
//     - model merging =
//        - Network exposes dictionary          // or use explicit expression new ConfigRecord(network)?
//        - ^^ + [ new nodes ] - [ nodes to delete ]
//          creates modified network
//        - pass into new NDLComputationNetwork
//     - also, any access needs to go up the chain and check for qualified matches there, and take the first
//       Or is that maybe the sole solution to the filter problem? [ ] + [ ] just computes a merged dict with possibly fully qualified names detected downstream?
//  - a way to explicitly access a symbol up from the current scope, needed for function parameters of the same name as dict entries created from them, e.g. the optional 'tag'
//     - ..X (e.g. ..tag)? Makes semi-sense, but syntactically easy, and hopefully not used too often
//     - or MACRO.X (e.g. Parameter.tag); latter would require to reference macros by name as a clearly defined mechanism, but hard to implement (ambiguity)
//  - name lookup should inject TextLocation into error stack
//  - doc strings for every parameter? E.g. LearnableParameter(rows{"Output dimension"},cols{"Input dimension"}) = new ...
//     - identifier become more complicated; they become a struct that carries the doc string
//  - expression-path problem:
//     - macro arg expressions get their path assigned when their thunk is created, the thunk remembers it
//     - however, really, the thunk should get the expression path from the context it is executed in, not the context it was created in
//     - maybe there is some clever scheme of overwriting when a result comes back? E.g. we retrieve a value but its name is not right, can we patch it up? Very tricky to find the right rules/conditions

}}} // namespaces
