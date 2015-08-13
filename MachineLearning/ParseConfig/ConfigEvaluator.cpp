// ConfigEvaluator.cpp -- execute what's given in a config file

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ConfigEvaluator.h"
#include <deque>
#include <set>
#include <functional>
#include <memory>
#include <cmath>

#ifndef let
#define let const auto
#endif

namespace Microsoft{ namespace MSR { namespace CNTK {

    using namespace std;
    using namespace msra::strfun;

    bool trace = true;      // enable to get debug output

#define exprPathSeparator L"."

    // =======================================================================
    // string formatting
    // =======================================================================

    wstring IndentString(wstring s, size_t indent)
    {
        const wstring prefix(indent, L' ');
        size_t pos = 0;
        for (;;)
        {
            s.insert(pos, prefix);
            pos = s.find(L'\n', pos + 2);
            if (pos == wstring::npos)
                return s;
            pos++;
        }
    }
    wstring NestString(wstring s, wchar_t open, bool newline, wchar_t close)
    {
        wstring result = IndentString(s, 2);
        if (newline)        // have a new line after the open symbol
            result = L" \n" + result + L"\n ";
        else
            result.append(L"  ");
        result.front() = open;
        result.back() = close;
        return result;
    }

    // 'how' is the center of a printf format string, without % and type. Example %.2f -> how=".2"
    // TODO: change to taking a regular format string and a :: array of args that are checked. Support d,e,f,g,x,c,s (s also for ToString()).
    // TODO: :: array. Check if that is the right operator for e.g. Haskell.
    // TODO: turn Print into PrintF; e.g. PrintF provides 'format' arg. Printf('solution to %s is %d', 'question' :: 42)
    static wstring FormatConfigValue(ConfigValuePtr arg, const wstring & how)
    {
        size_t pos = how.find(L'%');
        if (pos != wstring::npos)
            RuntimeError("FormatConfigValue: format string must not contain %");
        if (arg.Is<String>())
        {
            return wstrprintf((L"%" + how + L"s").c_str(), arg.AsRef<String>().c_str());
        }
        else if (arg.Is<Double>())
        {
            let val = arg.AsRef<Double>();
            if (val == (int)val)
                return wstrprintf((L"%" + how + L"d").c_str(), (int)val);
            else
                return wstrprintf((L"%" + how + L"f").c_str(), val);
        }
        else if (arg.Is<ConfigRecord>())
        {
            let record = arg.AsPtr<ConfigRecord>();
            let members = record->GetMembers();
            wstring result;
            bool first = true;
            for (auto iter : members)
            {
                if (first)
                    first = false;
                else
                    result.append(L"\n");
                result.append(iter.first);
                result.append(L" = ");
                result.append(FormatConfigValue(iter.second, how));
            }
            return NestString(result, L'[', true, L']');
        }
        else if (arg.Is<ConfigArray>())
        {
            let arr = arg.AsPtr<ConfigArray>();
            wstring result;
            let range = arr->GetRange();
            for (int i = range.first; i <= range.second; i++)
            {
                if (i > range.first)
                    result.append(L"\n");
                result.append(FormatConfigValue(arr->At(i, TextLocation()), how));
            }
            return NestString(result, L'(', false, L')');
        }
        else if (arg.Is<HasToString>())
            return arg.AsRef<HasToString>().ToString();
        else
            return msra::strfun::utf16(arg.TypeName());             // cannot print this type
    }

    // =======================================================================
    // support for late init  --currently broken
    // TODO: late init can be resolved at any assignment, no?
    //       As soon as the value we defer has a name, it has an object. Or maybe new! can only be assigned right away?
    // =======================================================================

    struct HasLateInit { virtual void Init(const ConfigRecord & config) = 0; }; // derive from this to indicate late initialization

    // =======================================================================
    // dummy implementation of several ComputationNode derivates for experimental purposes
    // =======================================================================

    struct Matrix { size_t rows; size_t cols; Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) { } };
    typedef shared_ptr<Matrix> MatrixPtr;

    struct HasName { virtual void SetName(const wstring & name) = 0; };

    set<wstring> nodesPrinted;      // HACK: ToString only formats nodes not already in here

    // TODO: should this expose a config dict to query the dimension (or only InputValues?)? Expose Children too? As list and by name?
    // TODO: constructor should take a vector of args in all cases.
    struct ComputationNode : public Object, public HasToString, public HasName
    {
        typedef shared_ptr<ComputationNode> ComputationNodePtr;

        // inputs and output
        vector<ComputationNodePtr> m_children;  // these are the inputs
        MatrixPtr m_functionValue;              // this is the result

        // other
        wstring m_nodeName;                     // node name in the graph
        static wstring TidyName(wstring name)
        {
            // clean out the intermediate name, e.g. A._b.C -> A.C for pretty printing of names, towards dictionary access
            // BUGBUG: anonymous ComputationNodes will get a non-unique name this way
            if (!name.empty())
            {
                let pos = name.find(exprPathSeparator);
                let left = pos == wstring::npos ? name : name.substr(0, pos);
                let right = pos == wstring::npos ? L"" : TidyName(name.substr(pos + 1));
                if (left.empty() || left[0] == '_')
                    name = right;
                else if (right.empty())
                    name = left;
                else
                    name = left + exprPathSeparator + right;
            }
            return name;
        }
        wstring NodeName() const { return m_nodeName; }        // TODO: should really be named GetNodeName()
        /*implement*/ void SetName(const wstring & name) { m_nodeName = name; }

        virtual const wchar_t * OperationName() const = 0;

        ComputationNode()
        {
            // node nmaes are not implemented yet; use a unique node name instead
            static int nodeIndex = 1;
            m_nodeName = wstrprintf(L"anonymousNode%d", nodeIndex);
            nodeIndex++;
        }

        virtual void AttachInputs(ComputationNodePtr arg)
        {
            m_children.resize(1);
            m_children[0] = arg;
        }
        virtual void AttachInputs(ComputationNodePtr leftNode, ComputationNodePtr rightNode)
        {
            m_children.resize(2);
            m_children[0] = leftNode;
            m_children[1] = rightNode;
        }
        virtual void AttachInputs(ComputationNodePtr arg1, ComputationNodePtr arg2, ComputationNodePtr arg3)
        {
            m_children.resize(3);
            m_children[0] = arg1;
            m_children[1] = arg2;
            m_children[2] = arg3;
        }

        /*implement*/ wstring ToString() const
        {
            // hack: remember we were already formatted
            // TODO: make nodesPrinted a static threadlocal member.
            //       Remember if we are first, and clear at end if so. Then it is not a hack anymore. Umm, won't work for Network though.
            let res = nodesPrinted.insert(NodeName());
            let alreadyPrinted = !res.second;
            if (alreadyPrinted)
                return TidyName(NodeName()) + L" ^";
            // we format it like "[TYPE] ( args )"
            wstring result = TidyName(NodeName()) + L" : " + wstring(OperationName());
            if (m_children.empty()) result.append(L"()");
            else
            {
                wstring args;
                bool first = true;
                for (auto & child : m_children)
                {
                    if (first)
                        first = false;
                    else
                        args.append(L"\n");
                    args.append(child->ToString());
                }
                result += L" " + NestString(args, L'(', true, ')');
            }
            return result;
        }
    };
    typedef ComputationNode::ComputationNodePtr ComputationNodePtr;
    class UnaryComputationNode : public ComputationNode
    {
    public:
        // TODO: how to inherit the base constructor? for derivates of this? constructor = default? using UnaryComputationNode::UnaryComputationNode
        UnaryComputationNode(ComputationNodePtr arg)
        {
            AttachInputs(arg);
        }
    };
    class BinaryComputationNode : public ComputationNode
    {
    public:
        BinaryComputationNode(ComputationNodePtr left, ComputationNodePtr right)
        {
            AttachInputs(left, right);
        }
    };
    class TernaryComputationNode : public ComputationNode
    {
    public:
        TernaryComputationNode(ComputationNodePtr arg1, ComputationNodePtr arg2, ComputationNodePtr arg3)
        {
            AttachInputs(arg1, arg2, arg3);
        }
    };

    class PlusNode : public BinaryComputationNode
    {
    public:
        PlusNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * OperationName() const { return L"Plus"; }
    };
    class MinusNode : public BinaryComputationNode
    {
    public:
        MinusNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * OperationName() const { return L"Minus"; }
    };
    class TimesNode : public BinaryComputationNode
    {
    public:
        TimesNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * OperationName() const { return L"Times"; }
    };
#if 0   // ScaleNode is something more complex it seems
    class ScaleNode : public ComputationNode
    {
        double factor;
    public:
        TimesNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * OperationName() const { return L"Scale"; }
    };
#endif
    class LogNode : public UnaryComputationNode
    {
    public:
        LogNode(ComputationNodePtr arg) : UnaryComputationNode(arg) { }
        /*implement*/ const wchar_t * OperationName() const { return L"Log"; }
    };
    class SigmoidNode : public UnaryComputationNode
    {
    public:
        SigmoidNode(ComputationNodePtr arg) : UnaryComputationNode(arg) { }
        /*implement*/ const wchar_t * OperationName() const { return L"Sigmoid"; }
    };
    class MeanNode : public UnaryComputationNode
    {
    public:
        MeanNode(ComputationNodePtr arg) : UnaryComputationNode(arg) { }
        /*implement*/ const wchar_t * OperationName() const { return L"Mean"; }
    };
    class InvStdDevNode : public UnaryComputationNode
    {
    public:
        InvStdDevNode(ComputationNodePtr arg) : UnaryComputationNode(arg) { }
        /*implement*/ const wchar_t * OperationName() const { return L"InvStdDev"; }
    };
    class PerDimMeanVarNormalizationNode : public TernaryComputationNode
    {
    public:
        PerDimMeanVarNormalizationNode(ComputationNodePtr arg1, ComputationNodePtr arg2, ComputationNodePtr arg3) : TernaryComputationNode(arg1, arg2, arg3) { }
        /*implement*/ const wchar_t * OperationName() const { return L"PerDimMeanVarNormalization"; }
    };
    class RowSliceNode : public UnaryComputationNode
    {
        size_t firstRow, numRows;
    public:
        RowSliceNode(ComputationNodePtr arg, size_t firstRow, size_t numRows) : UnaryComputationNode(arg), firstRow(firstRow), numRows(numRows) { }
        /*implement*/ const wchar_t * OperationName() const { return L"RowSlice"; }
    };
    class CrossEntropyWithSoftmaxNode : public BinaryComputationNode
    {
    public:
        CrossEntropyWithSoftmaxNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * OperationName() const { return L"CrossEntropyWithSoftmax"; }
    };
    class ErrorPredictionNode : public BinaryComputationNode
    {
    public:
        ErrorPredictionNode(ComputationNodePtr left, ComputationNodePtr right) : BinaryComputationNode(left, right) { }
        /*implement*/ const wchar_t * OperationName() const { return L"ErrorPrediction"; }
    };
    // BROKEN
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
        /*implement*/ const wchar_t * OperationName() const { return L"Delay"; }
    };
    class InputValue : public ComputationNode
    {
    public:
        InputValue(const ConfigRecord & config) // TODO
        {
            config;
        }
        /*implement*/ const wchar_t * OperationName() const { return L"InputValue"; }
    };
    class LearnableParameter : public ComputationNode
    {
        size_t outDim, inDim;
    public:
        LearnableParameter(size_t outDim, size_t inDim) : outDim(outDim), inDim(inDim) { }
        /*implement*/ const wchar_t * OperationName() const { return L"LearnableParameter"; }
        /*implement*/ wstring ToString() const
        {
            let res = nodesPrinted.insert(NodeName());
            let alreadyPrinted = !res.second;
            if (alreadyPrinted)
                return TidyName(NodeName()) + L" ^";
            else
                return wstrprintf(L"%ls : %ls (%d, %d)", TidyName(NodeName()).c_str(), OperationName(), (int)outDim, (int)inDim);
        }
    };
    // factory function for ComputationNodes
    template<>
    shared_ptr<ComputationNode> MakeRuntimeObject<ComputationNode>(const ConfigRecord & config)
    {
        let classIdParam = config[L"class"];
        wstring classId = classIdParam;
        if (classId == L"LearnableParameterNode")
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
        else if (classId == L"LogNode")
            return make_shared<LogNode>((ComputationNodePtr)config[L"arg"]);
        else if (classId == L"SigmoidNode")
            return make_shared<SigmoidNode>((ComputationNodePtr)config[L"arg"]);
        else if (classId == L"MeanNode")
            return make_shared<MeanNode>((ComputationNodePtr)config[L"arg"]);
        else if (classId == L"InvStdDevNode")
            return make_shared<InvStdDevNode>((ComputationNodePtr)config[L"arg"]);
        else if (classId == L"PerDimMeanVarNormalizationNode")
            return make_shared<PerDimMeanVarNormalizationNode>((ComputationNodePtr)config[L"arg1"], (ComputationNodePtr)config[L"arg2"], (ComputationNodePtr)config[L"arg3"]);
        else if (classId == L"RowSliceNode")
            return make_shared<RowSliceNode>((ComputationNodePtr)config[L"arg"], (size_t)config[L"first"], (size_t)config[L"num"]);
        else if (classId == L"CrossEntropyWithSoftmaxNode")
            return make_shared<CrossEntropyWithSoftmaxNode>((ComputationNodePtr)config[L"left"], (ComputationNodePtr)config[L"right"]);
        else if (classId == L"ErrorPredictionNode")
            return make_shared<ErrorPredictionNode>((ComputationNodePtr)config[L"left"], (ComputationNodePtr)config[L"right"]);
        throw EvaluationError(L"unknown ComputationNode class " + classId, classIdParam.GetLocation());
    }

    // =======================================================================
    // dummy implementations of Network derivates
    // =======================================================================

    // Network class
    class Network : public Object, public IsConfigRecord
    {
    public:
        // pretending to be a ConfigRecord
        /*implement*/ const ConfigValuePtr & operator[](const wstring & id) const   // e.g. confRec[L"message"]
        {
            id;  RuntimeError("unknown class parameter");    // (for now)
        }
        /*implement*/ const ConfigValuePtr * Find(const wstring & id) const         // returns nullptr if not found
        {
            id;  return nullptr; // (for now)
        }
    };

    class NDLNetwork : public Network, public HasToString
    {
        set<ComputationNodePtr> nodes;  // root nodes in this network; that is, nodes defined in the dictionary
    public:
        NDLNetwork(const ConfigRecord & config)
        {
            // we collect all ComputationNodes from the config; that's it
            let members = config.GetMembers();
            for (auto iter : members)
            {
                if (!iter.second.Is<ComputationNode>())
                    continue;
                nodes.insert((ComputationNodePtr)config[iter.first]);
            }
        }
        /*implement*/ wstring ToString() const
        {
            // hack: remember we were already formatted
            nodesPrinted.clear();
            // print all nodes we got
            wstring args;
            bool first = true;
            for (auto & node : nodes)
            {
                if (first)
                    first = false;
                else
                    args.append(L"\n");
                args.append(node->ToString());
            }
            return L"NDLNetwork " + NestString(args, L'[', true, ']');
        }
    };

    // =======================================================================
    // built-in functions (implemented as Objects that are also their value)
    // =======================================================================

    // sample objects to implement functions
    // TODO: Chr(), Substr(), Replace(), RegexReplace()     Substr takes negative position to index from end, and length -1
    // TODO: NumericFunctions: Floor(), Ceil(), Round(), Length()     (make Abs, Sign, Min and Max macros; maybe also Ceil=-Floor(-x) and Round=Floor(x+0.5)!)
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

    // =======================================================================
    // general-purpose use Actions
    // =======================================================================

    // sample runtime objects for testing
    // We are trying all sorts of traits here, even if they make no sense for PrintAction.
    class PrintAction : public Object, public HasLateInit, public HasName
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
        /*implement*/ void SetName(const wstring & name)
        {
            name;
        }
    };

    class AnotherAction : public Object
    {
    public:
        AnotherAction(const ConfigRecord &) { fprintf(stderr, "Another\n"); }
        virtual ~AnotherAction(){}
    };

    // =======================================================================
    // Evaluator -- class for evaluating a syntactic parse tree
    // Evaluation converts a parse tree from ParseConfigString/File() into a graph of live C++ objects.
    // TODO: This class has no members except for pre-initialized lookup tables. We could get rid of the class.
    // =======================================================================

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
        // -----------------------------------------------------------------------
        // error handling
        // -----------------------------------------------------------------------

        __declspec(noreturn) void Fail(const wstring & msg, TextLocation where) { throw EvaluationError(msg, where); }

        __declspec(noreturn) void TypeExpected(const wstring & what, ExpressionPtr e) { Fail(L"expected expression of type " + what, e->location); }
        __declspec(noreturn) void UnknownIdentifier(const wstring & id, TextLocation where) { Fail(L"unknown member name " + id, where); }

        // -----------------------------------------------------------------------
        // lexical scope
        // -----------------------------------------------------------------------

        struct Scope
        {
            shared_ptr<ConfigRecord> symbols;   // symbols in this scope
            shared_ptr<Scope> up;               // one scope up
            Scope(shared_ptr<ConfigRecord> symbols, shared_ptr<Scope> up) : symbols(symbols), up(up) { }
        };
        typedef shared_ptr<Scope> ScopePtr;
        ScopePtr MakeScope(shared_ptr<ConfigRecord> symbols, shared_ptr<Scope> up) { return make_shared<Scope>(symbols, up); }

        // -----------------------------------------------------------------------
        // configurable runtime types ("new" expression)
        // -----------------------------------------------------------------------

        // helper for configurableRuntimeTypes initializer below
        // This returns a lambda that is a constructor for a given runtime type, and a bool saying whether T derives from IsConfigRecord.
        // LateInit currently broken.
        template<class C>
        pair<function<ConfigValuePtr(const ConfigRecord &, TextLocation)>,bool> MakeRuntimeTypeConstructor()
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
                let lambda = [this](const ConfigRecord & config, TextLocation location)
                {
                    return ConfigValuePtr(MakeRuntimeObject<C>(config), location);
                };
            let isConfigRecord = is_base_of<IsConfigRecord, C>::value;
            return make_pair(lambda, isConfigRecord);
        }
        // initialize the lookup table
        void InitConfigurableRuntimeTypes()
        {
#define DefineRuntimeType(T) { L#T, MakeRuntimeTypeConstructor<T>() }
            // TODO: add a second entry that tests whether T derives from IsConfigRecord. Or MakeRuntimeTypeConstructor could return a std::pair.
            // lookup table for "new" expression
            configurableRuntimeTypes = decltype(configurableRuntimeTypes)
            {
                // ComputationNodes
                DefineRuntimeType(ComputationNode),
                // other relevant classes
                DefineRuntimeType(NDLNetwork),
                // Functions
                DefineRuntimeType(StringFunction),
                // Actions
                DefineRuntimeType(PrintAction),
                DefineRuntimeType(AnotherAction),
            };
        }

        // -----------------------------------------------------------------------
        // late initialization   --currently broken
        // -----------------------------------------------------------------------

        // "new!" expressions get queued for execution after all other nodes of tree have been executed
        // TODO: This is totally broken, need to figuree out the deferred process first.
        struct LateInitItem
        {
            ConfigValuePtr object;
            ScopePtr scope;
            ExpressionPtr dictExpr;                             // the dictionary expression that now can be fully evaluated
            LateInitItem(ConfigValuePtr object, ScopePtr scope, ExpressionPtr dictExpr) : object(object), scope(scope), dictExpr(dictExpr) { }
        };

        // -----------------------------------------------------------------------
        // name lookup
        // -----------------------------------------------------------------------

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

        // look up an identifier in an expression that is a ConfigRecord
        ConfigValuePtr RecordLookup(ExpressionPtr recordExpr, const wstring & id, TextLocation idLocation, ScopePtr scope, const wstring & exprPath)
        {
            let record = AsPtr<ConfigRecord>(Evaluate(recordExpr, scope, exprPath, L""), recordExpr, L"record");
            return ResolveIdentifier(id, idLocation, MakeScope(record, nullptr/*no up scope*/));
        }

        // -----------------------------------------------------------------------
        // runtime-object creation
        // -----------------------------------------------------------------------

        // evaluate all elements in a dictionary expression and turn that into a ConfigRecord
        // which is meant to be passed to the constructor or Init() function of a runtime object
        shared_ptr<ConfigRecord> ConfigRecordFromDictExpression(ExpressionPtr recordExpr, ScopePtr scope, const wstring & exprPath)
        {
            // evaluate the record expression itself
            // This will leave its members unevaluated since we do that on-demand
            // (order and what gets evaluated depends on what is used).
            let record = AsPtr<ConfigRecord>(Evaluate(recordExpr, scope, exprPath, L""), recordExpr, L"record");
            // resolve all entries, as they need to be passed to the C++ world which knows nothing about this
            record->ResolveAll();
            return record;
        }

        // perform late initialization
        // This assumes that the ConfigValuePtr points to a BoxWithLateInitOf. If not, it will fail with a nullptr exception.
        void LateInit(LateInitItem & lateInitItem)
        {
            let config = ConfigRecordFromDictExpression(lateInitItem.dictExpr, lateInitItem.scope, L""/*BROKEN*/);
            let object = lateInitItem.object;
            auto p = object.AsRef<shared_ptr<HasLateInit>>();  // TODO: AsPtr?
            p->Init(*config);
//            dynamic_cast<HasLateInit*>(lateInitItem.object.get())->Init(*config);  // call BoxWithLateInitOf::Init() which in turn will call HasLateInite::Init() on the actual object
        }

        // -----------------------------------------------------------------------
        // access to ConfigValuePtr content with error messages
        // -----------------------------------------------------------------------

        // get value
        template<typename T>
        shared_ptr<T> AsPtr(ConfigValuePtr value, ExpressionPtr e, const wchar_t * typeForMessage)
        {
            if (!value.Is<T>())
                TypeExpected(typeForMessage, e);
            return value.AsPtr<T>();
        }

        double ToDouble(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = dynamic_cast<Double*>(value.get());
            if (!val)
                TypeExpected(L"number", e);
            double & dval = *val;
            return dval;    // great place to set breakpoint
        }

        // get number and return it as an integer (fail if it is fractional)
        int ToInt(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = ToDouble(value, e);
            let res = (int)(val);
            if (val != res)
                TypeExpected(L"integer", e);
            return res;
        }

        bool ToBoolean(ConfigValuePtr value, ExpressionPtr e)
        {
            let val = dynamic_cast<Bool*>(value.get());            // TODO: factor out this expression
            if (!val)
                TypeExpected(L"boolean", e);
            return *val;
        }

        // -----------------------------------------------------------------------
        // infix operators
        // -----------------------------------------------------------------------

        typedef function<ConfigValuePtr(ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const wstring & exprPath)> InfixFunction;
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

        // evaluate a Boolean expression (all types)
        template<typename T>
        ConfigValuePtr CompOp(ExpressionPtr e, const T & left, const T & right)
        {
            if (e->op == L"==")      return MakePrimitiveConfigValuePtr(left == right, e->location);
            else if (e->op == L"!=") return MakePrimitiveConfigValuePtr(left != right, e->location);
            else if (e->op == L"<")  return MakePrimitiveConfigValuePtr(left <  right, e->location);
            else if (e->op == L">")  return MakePrimitiveConfigValuePtr(left >  right, e->location);
            else if (e->op == L"<=") return MakePrimitiveConfigValuePtr(left <= right, e->location);
            else if (e->op == L">=") return MakePrimitiveConfigValuePtr(left >= right, e->location);
            else LogicError("unexpected infix op");
        }

        // directly instantiate a ComputationNode for the magic operators * + and - that are automatically translated.
        ConfigValuePtr MakeMagicComputationNode(const wstring & classId, TextLocation location, const ConfigValuePtr & left, const ConfigValuePtr & right,
                                                const wstring & exprPath)
        {
            // find creation lambda
            let newIter = configurableRuntimeTypes.find(L"ComputationNode");
            if (newIter == configurableRuntimeTypes.end())
                LogicError("unknown magic runtime-object class");
            // form the ConfigRecord
            ConfigRecord config;
            config.Add(L"class", location, ConfigValuePtr(make_shared<String>(classId), location));
            config.Add(L"left", left.GetLocation(), left);
            config.Add(L"right", right.GetLocation(), right);
            // instantiate
            let value = newIter->second.first(config, location);
            let valueWithName = dynamic_cast<HasName*>(value.get());
            if (valueWithName && !exprPath.empty())
                valueWithName->SetName(exprPath);
            return value;
        }

        // initialize the infixOps table
        void InitInfixOps()
        {
            // lookup table for infix operators
            // helper lambdas for evaluating infix operators
            InfixFunction NumOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const wstring & /*exprPath*/) -> ConfigValuePtr
            {
                let left  = leftVal.AsRef<Double>();
                let right = rightVal.AsRef<Double>();
                if (e->op == L"+")       return MakePrimitiveConfigValuePtr(left + right, e->location);
                else if (e->op == L"-")  return MakePrimitiveConfigValuePtr(left - right, e->location);
                else if (e->op == L"*")  return MakePrimitiveConfigValuePtr(left * right, e->location);
                else if (e->op == L"/")  return MakePrimitiveConfigValuePtr(left / right, e->location);
                else if (e->op == L"%")  return MakePrimitiveConfigValuePtr(fmod(left, right), e->location);
                else if (e->op == L"**") return MakePrimitiveConfigValuePtr(pow(left, right), e->location);
                else return CompOp<double> (e, left, right);
            };
            InfixFunction StrOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const wstring & /*exprPath*/) -> ConfigValuePtr
            {
                let left  = leftVal.AsRef<String>();
                let right = rightVal.AsRef<String>();
                if (e->op == L"+")  return ConfigValuePtr(make_shared<String>(left + right), e->location);
                else return CompOp<wstring>(e, left, right);
            };
            InfixFunction BoolOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const wstring & /*exprPath*/) -> ConfigValuePtr
            {
                let left  = leftVal.AsRef<Bool>();
                let right = rightVal.AsRef<Bool>();
                if (e->op == L"||")       return MakePrimitiveConfigValuePtr(left || right, e->location);
                else if (e->op == L"&&")  return MakePrimitiveConfigValuePtr(left && right, e->location);
                else if (e->op == L"^")   return MakePrimitiveConfigValuePtr(left ^  right, e->location);
                else return CompOp<bool>(e, left, right);
            };
            InfixFunction NodeOp = [this](ExpressionPtr e, ConfigValuePtr leftVal, ConfigValuePtr rightVal, const wstring & exprPath) -> ConfigValuePtr
            {
                // TODO: test this
                if (rightVal.Is<Double>())     // ComputeNode * scalar
                    swap(leftVal, rightVal);        // -> scalar * ComputeNode
                if (leftVal.Is<Double>())      // scalar * ComputeNode
                {
                    if (e->op == L"*")  return MakeMagicComputationNode(L"ScaleNode", e->location, leftVal, rightVal, exprPath);
                    else LogicError("unexpected infix op");
                }
                else                                // ComputeNode OP ComputeNode
                {
                    if (e->op == L"+")       return MakeMagicComputationNode(L"PlusNode",  e->location, leftVal, rightVal, exprPath);
                    else if (e->op == L"-")  return MakeMagicComputationNode(L"MinusNode", e->location, leftVal, rightVal, exprPath);
                    else if (e->op == L"*")  return MakeMagicComputationNode(L"TimesNode", e->location, leftVal, rightVal, exprPath);
                    // TODO: forgot DiagTimes()
                    else LogicError("unexpected infix op");
                }
            };
            InfixFunction BadOp = [this](ExpressionPtr e, ConfigValuePtr, ConfigValuePtr, const wstring &) -> ConfigValuePtr { FailBinaryOpTypes(e); };
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

        // -----------------------------------------------------------------------
        // thunked (delayed) evaluation
        // -----------------------------------------------------------------------

        // create a lambda that calls Evaluate() on an expr to get or realize its value
        shared_ptr<ConfigValuePtr::Thunk> MakeEvaluateThunkPtr(ExpressionPtr expr, ScopePtr scope, const wstring & exprPath, const wstring & exprId)
        {
            function<ConfigValuePtr()> f = [this, expr, scope, exprPath, exprId]()   // lambda that computes this value of 'expr'
            {
                if (trace)
                    expr->location.PrintIssue(L"", exprPath.c_str(), L"executing thunk");
                let value = Evaluate(expr, scope, exprPath, exprId);
                return value;   // this is a great place to set a breakpoint!
            };
            return make_shared<ConfigValuePtr::Thunk>(f, expr->location);
        }

        // -----------------------------------------------------------------------
        // lookup tables
        // -----------------------------------------------------------------------

        // all infix operators with lambdas for evaluating them
        map<wstring, InfixFunctions> infixOps;

        // this table lists all C++ types that can be instantiated from "new" expressions
        // The pair contains a lambda and a bool indicating whether the class derives from IsConfigRecord (which, if so, would reset exprPath).
        map<wstring, pair<function<ConfigValuePtr(const ConfigRecord &, TextLocation)>,bool>> configurableRuntimeTypes;

        // -----------------------------------------------------------------------
        // main evaluator function (highly recursive)
        // -----------------------------------------------------------------------

        // Evaluate()
        //  - input:  expression
        //  - output: ConfigValuePtr that holds the evaluated value of the expression
        // Note that returned values may include complex value types like dictionaries (ConfigRecord) and functions (ConfigLambda).
        ConfigValuePtr Evaluate(ExpressionPtr e, ScopePtr scope, wstring exprPath, const wstring & exprId)
        {
            // expression names
            // Merge exprPath and exprId into one unless one is empty
            if (!exprPath.empty() && !exprId.empty())
                exprPath.append(exprPathSeparator);
            exprPath.append(exprId);
            // tracing
            if (trace)
                e->location.PrintIssue(L"", L"", L"trace");
            // --- literals
            if (e->op == L"d")       return MakePrimitiveConfigValuePtr(e->d, e->location);         // === double literal
            else if (e->op == L"s")  return ConfigValuePtr(make_shared<String>(e->s), e->location); // === string literal
            else if (e->op == L"b")  return MakePrimitiveConfigValuePtr(e->b, e->location);         // === bool literal
            else if (e->op == L"new" || e->op == L"new!")                                           // === 'new' expression: instantiate C++ runtime object
            {
                // find the constructor lambda
                let newIter = configurableRuntimeTypes.find(e->id);
                if (newIter == configurableRuntimeTypes.end())
                    Fail(L"unknown runtime type " + e->id, e->location);
                // form the config record
                let dictExpr = e->args[0];
                ConfigValuePtr value;
                let argsExprPath = newIter->second.second ? L"" : exprPath;   // reset expr-name path if object exposes a dictionary
                if (e->op == L"new")   // evaluate the parameter dictionary into a config record
                    value = newIter->second.first(*ConfigRecordFromDictExpression(dictExpr, scope, argsExprPath), e->location); // this constructs it
                else                // ...unless it's late init. Then we defer initialization.
                {
                    // TODO: need a check here whether the class allows late init, before we actually try, so that we can give a concise error message
                    // ... exprName broken
                    // TODO: allow "new!" only directly after an assignment, and make that assignment delayed
                    let value = newIter->second.first(ConfigRecord(), e->location);
                    deferredInitList.push_back(LateInitItem(value, scope, dictExpr)); // construct empty and remember to Init() later
                }
                let valueWithName = dynamic_cast<HasName*>(value.get());
                if (valueWithName && !exprPath.empty())
                    valueWithName->SetName(exprPath);
                return value;   // we return the created but not initialized object as the value, so others can reference it
            }
            else if (e->op == L"if")                                                    // === conditional expression
            {
                let condition = ToBoolean(Evaluate(e->args[0], scope, exprPath, L"_if"), e->args[0]);
                if (condition)
                    return Evaluate(e->args[1], scope, exprPath, L"_then");   // TODO: pass exprName through 'if'?
                else
                    return Evaluate(e->args[2], scope, exprPath, L"_else");
            }
            // --- functions
            else if (e->op == L"=>")                                                    // === lambda (all macros are stored as lambdas)
            {
                // on scope: The lambda expression remembers the lexical scope of the '=>'; this is how it captures its context.
                let argListExpr = e->args[0];           // [0] = argument list ("()" expression of identifiers, possibly optional args)
                if (argListExpr->op != L"()") LogicError("parameter list expected");
                let fnExpr = e->args[1];                // [1] = expression of the function itself
                let f = [this, argListExpr, fnExpr, scope, exprPath](const vector<ConfigValuePtr> & args, const shared_ptr<ConfigRecord> & namedArgs, const wstring & callerExprPath) -> ConfigValuePtr
                {
                    // on exprName
                    //  - 'callerExprPath' is the name to which the result of the fn evaluation will be assigned
                    //  - 'exprPath' (outside) is the name of the macro we are defining this lambda under
                    let & argList = argListExpr->args;
                    if (args.size() != argList.size()) LogicError("function application with mismatching number of arguments");
                    // create a ConfigRecord with param names from 'argList' and values from 'args'
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
                    // get the macro name for the exprPath
                    wstring macroId = exprPath;
                    let pos = macroId.find(exprPathSeparator);
                    if (pos != wstring::npos)
                        macroId.erase(0, pos + 1);
                    // now evaluate the function
                    return Evaluate(fnExpr, MakeScope(record, scope), callerExprPath, L"_[" + macroId + L"]");  // bring args into scope; keep lex scope of '=>' as upwards chain
                };
                let record = make_shared<ConfigRecord>();   // TODO: named args go here
                return ConfigValuePtr(make_shared<ConfigLambda>(argListExpr->args.size(), record, f), e->location);
            }
            else if (e->op == L"(")                                         // === apply a function to its arguments
            {
                let lambdaExpr = e->args[0];            // [0] = function
                let argsExpr = e->args[1];              // [1] = arguments passed to the function ("()" expression of expressions)
                let lambda = AsPtr<ConfigLambda>(Evaluate(lambdaExpr, scope, exprPath, L"_lambda"), lambdaExpr, L"function");
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
                    argVals[i] = ConfigValuePtr(MakeEvaluateThunkPtr(argValExpr, scope, exprPath, wstrprintf(L"_arg%d", i)), argValExpr->location);  // make it a thunked value
                    /*this wstrprintf should be gone, this is now the exprName*/
                }
                // deal with namedArgs later
                let namedArgs = make_shared<ConfigRecord>();
                // call the function!
                return lambda->Apply(argVals, namedArgs, exprPath);
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
                    let id = entry.first;
                    let expr = entry.second.second;                 // expression to compute the entry
                    record->Add(id, entry.second.first/*loc of id*/, ConfigValuePtr(MakeEvaluateThunkPtr(expr, thisScope, exprPath, id), expr->location));
                }
                // BUGBUG: wrong text location passed in. Should be the one of the identifier, not the RHS. NamedArgs have no location.
                return ConfigValuePtr(record, e->location);
            }
            else if (e->op == L"id") return ResolveIdentifier(e->id, e->location, scope);   // === variable/macro access within current scope
            else if (e->op == L".")                                                 // === variable/macro access in given ConfigRecord element
            {
                let recordExpr = e->args[0];
                return RecordLookup(recordExpr, e->id, e->location, scope, exprPath);
            }
            // --- arrays
            else if (e->op == L":")                                                 // === array expression (-> ConfigArray)
            {
                // this returns a flattened list of all members as a ConfigArray type
                let arr = make_shared<ConfigArray>();       // note: we could speed this up by keeping the left arg and appending to it
                for (size_t i = 0; i < e->args.size(); i++) // concatenate the two args
                {
                    let expr = e->args[i];
                    let item = Evaluate(expr, scope, exprPath, wstrprintf(L"_vecelem%d", i));           // result can be an item or a vector
                    if (item.Is<ConfigArray>())
                        arr->Append(item.AsRef<ConfigArray>());     // append all elements (this flattens it)
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
                let firstIndex = ToInt(Evaluate(firstIndexExpr, scope, exprPath, L"_first"), firstIndexExpr);
                let lastIndex  = ToInt(Evaluate(lastIndexExpr,  scope, exprPath, L"_last"),  lastIndexExpr);
                let lambda = AsPtr<ConfigLambda>(Evaluate(initLambdaExpr, scope, exprPath, L"_initializer"), initLambdaExpr, L"function");
                if (lambda->GetNumParams() != 1)
                    Fail(L"'array' requires an initializer function with one argument (the index)", initLambdaExpr->location);
                // At this point, we must know the dimensions and the initializer lambda, but we don't need to know all array elements.
                // Resolving array members on demand allows recursive access to the array variable, e.g. h[t] <- f(h[t-1]).
                // create a vector of Thunks to initialize each value
                vector<ConfigValuePtr> elementThunks;
                for (int index = firstIndex; index <= lastIndex; index++)
                {
                    let indexValue = MakePrimitiveConfigValuePtr((double)index, e->location);           // index as a ConfigValuePtr
                    let elemExprPath = exprPath.empty() ? L"" : wstrprintf(L"%ls[%d]", exprPath.c_str(), index);    // expression name shows index lookup
                    // create an expression
                    function<ConfigValuePtr()> f = [this, indexValue, initLambdaExpr, scope, elemExprPath]()   // lambda that computes this value of 'expr'
                    {
                        if (trace)
                            initLambdaExpr->location.PrintIssue(L"", wstrprintf(L"index %d", (int)indexValue).c_str(), L"executing array initializer thunk");
                        // apply initLambdaExpr to indexValue and return the resulting value
                        let initLambda = AsPtr<ConfigLambda>(Evaluate(initLambdaExpr, scope, elemExprPath, L""), initLambdaExpr, L"function");
                        vector<ConfigValuePtr> argVals(1, indexValue);  // create an arg list with indexValue as the one arg
                        let namedArgs = make_shared<ConfigRecord>();    // no named args in initializer lambdas
                        let value = initLambda->Apply(argVals, namedArgs, elemExprPath);
                        return value;   // this is a great place to set a breakpoint!
                    };
                    elementThunks.push_back(ConfigValuePtr(make_shared<ConfigValuePtr::Thunk>(f, initLambdaExpr->location), initLambdaExpr->location));
                }
                auto arr = make_shared<ConfigArray>(firstIndex, move(elementThunks));
                return ConfigValuePtr(arr, e->location);
            }
            else if (e->op == L"[")                                         // === access array element by index
            {
                let arrValue = Evaluate(e->args[0], scope, exprPath, L"_vector");
                let indexExpr = e->args[1];
                let arr = AsPtr<ConfigArray>(arrValue, indexExpr, L"array");
                let index = ToInt(Evaluate(indexExpr, scope, exprPath, L"_index"), indexExpr);
                return arr->At(index, indexExpr->location);
            }
            // --- unary operators '+' '-' and '!'
            else if (e->op == L"+(" || e->op == L"-(")                      // === unary operators + and -
            {
                let argExpr = e->args[0];
                let argValPtr = Evaluate(argExpr, scope, exprPath, e->op == L"+(" ? L"" : L"_negate");
                if (argValPtr.Is<Double>())
                    if (e->op == L"+(") return argValPtr;
                    else return MakePrimitiveConfigValuePtr(-(double)argValPtr, e->location);
                else if (argValPtr.Is<ComputationNode>())   // -ComputationNode becomes ScaleNode(-1,arg)
                    if (e->op == L"+(") return argValPtr;
                    else return MakeMagicComputationNode(L"ScaleNode", e->location, MakePrimitiveConfigValuePtr(-1.0, e->location), argValPtr, exprPath);
                else
                    Fail(L"operator '" + e->op.substr(0, 1) + L"' cannot be applied to this operand", e->location);
            }
            else if (e->op == L"!(")                                        // === unary operator !
            {
                let arg = ToBoolean(Evaluate(e->args[0], scope, exprPath, L"_not"), e->args[0]);
                return MakePrimitiveConfigValuePtr(!arg, e->location);
            }
            // --- regular infix operators such as '+' and '=='
            else
            {
                let opIter = infixOps.find(e->op);
                if (opIter == infixOps.end())
                    LogicError("e->op " + utf8(e->op) + " not implemented");
                let & functions = opIter->second;
                let leftArg = e->args[0];
                let rightArg = e->args[1];
                let leftValPtr  = Evaluate(leftArg,  scope, exprPath, L"_op0");
                let rightValPtr = Evaluate(rightArg, scope, exprPath, L"_op1");
                if (leftValPtr.Is<Double>() && rightValPtr.Is<Double>())
                    return functions.NumbersOp(e, leftValPtr, rightValPtr, exprPath);
                else if (leftValPtr.Is<String>() && rightValPtr.Is<String>())
                    return functions.StringsOp(e, leftValPtr, rightValPtr, exprPath);
                else if (leftValPtr.Is<Bool>() && rightValPtr.Is<Bool>())
                    return functions.BoolOp(e, leftValPtr, rightValPtr, exprPath);
                // ComputationNode is "magic" in that we map *, +, and - to know classes of fixed names.
                else if (leftValPtr.Is<ComputationNode>() && rightValPtr.Is<ComputationNode>())
                    return functions.ComputeNodeOp(e, leftValPtr, rightValPtr, exprPath);
                else if (leftValPtr.Is<ComputationNode>() && rightValPtr.Is<Double>())
                    return functions.ComputeNodeNumberOp(e, leftValPtr, rightValPtr, exprPath);
                else if (leftValPtr.Is<Double>() && rightValPtr.Is<ComputationNode>())
                    return functions.NumberComputeNodeOp(e, leftValPtr, rightValPtr, exprPath);
                // TODO: DictOp
                else
                    FailBinaryOpTypes(e);
            }
            //LogicError("should not get here");
        }

        // Traverse through the expression (parse) tree to evaluate a value.    --TODO broken
        deque<LateInitItem> deferredInitList;
    public:
        // -----------------------------------------------------------------------
        // constructor
        // -----------------------------------------------------------------------

        Evaluator()
        {
            InitConfigurableRuntimeTypes();
            InitInfixOps();
        }

        // TODO: deferred list not working at all.
        //       Do() just calls into EvaluateParse directly.
        //       Need to move this list into Evaluate() directly and figure it out.
        ConfigValuePtr EvaluateParse(ExpressionPtr e)
        {
            auto result = Evaluate(e, nullptr/*top scope*/, L"", L"$");
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
            RecordLookup(e, L"do", e->location, nullptr, L"$");  // we evaluate the member 'do'
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
