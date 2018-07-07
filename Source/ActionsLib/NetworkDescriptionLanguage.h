//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "ComputationNetwork.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// EqualInsensitive - check to see if two nodes are equal up to the length of the first string (must be at least half as long as actual node name)
// string1 - [in,out] string to compare, if comparision is equal insensitive but not sensitive, will replace with sensitive version
// string2 - second string to compare
// alternate - alternate naming of the string
// return - true if strings are equal insensitive and modifies string1 to sensitive version if different
bool EqualInsensitive(std::wstring& string1, const std::wstring& string2, const wchar_t* alternate = NULL);

// CheckFunction - check to see if we match a function name
// string1 - [in,out] string to compare, if comparision is equal and at least half the full node name will replace with full node name
// allowUndeterminedVariable - [out] set to true if undetermined variables (symbols yet to be defined) are allowed here
// return - true if function name found
bool CheckFunction(std::string& p_nodeType, bool* allowUndeterminedVariable = nullptr);

// NDLType - Network Description Language node type
enum NDLType
{
    ndlTypeNull,
    ndlTypeConstant,
    ndlTypeFunction,
    ndlTypeVariable,
    ndlTypeParameter,    // parameter value, must be looked up to get actual value
    ndlTypeUndetermined, // an undetermined value that will later be resolved
    ndlTypeOptionalParameter,
    ndlTypeArray,
    ndlTypeMacroCall, // calling a macro
    ndlTypeMacro,     // definition of a macro
    ndlTypeMax
};

// NDLPass - enumeration for the number of passes through the NDL parser
enum NDLPass
{
    ndlPassInitial,            // inital pass, create nodes
    ndlPassResolve,            // resolve any undetermined symbols (variables that were not yet declared in NDL)
    ndlPassFinal,              // final pass done post-validation (when all matrices are allocated to the correct size)
    ndlPassAll = ndlPassFinal, // all passes, used as flag in NDLUtil.h
    ndlPassMax                 // number of NDLPasses
};

// ++ operator for this enum, so loops work
NDLPass& operator++(NDLPass& ndlPass);

// Predeclaration of Script and Node
template <typename ElemType>
class NDLScript;

template <typename ElemType>
class NDLNode;

// NDLNodeEvaluator - Node evaluation interface
// implemented by execution engines to convert script to approriate internal formats
template <typename ElemType>
class NDLNodeEvaluator
{
public:
    virtual void Evaluate(NDLNode<ElemType>* node, const wstring& baseName, const NDLPass pass) = 0;
    virtual ~NDLNodeEvaluator() = 0;

    // EvaluateParameter - Evaluate a parameter of a call
    // node - NDLNode of the script
    // nodeParam - NDLNode parameter we are evaluating
    // baseName - name of the base node
    // pass - which pass through the NDL nodes
    // returns: the node that is the evaluated parameter
    virtual NDLNode<ElemType>* EvaluateParameter(NDLNode<ElemType>* node, NDLNode<ElemType>* nodeParam, const std::wstring& baseName, const NDLPass pass) = 0;

    // EvaluateParameters - Evaluate the parameters of a call
    // node - NDLNode we are evaluating parameters for
    // baseName - baseName for the current node
    // nodeParamStart - starting parameter that contains a node
    // nodeParamCount - ending parameter that contains a node
    // pass - NDL pass we are evaluating
    // returns: vector of eval pointers, which are ComputationNodePtr for CNEvaluator
    virtual std::vector<void*> EvaluateParameters(NDLNode<ElemType>* node, const wstring& baseName, int nodeParamStart, int nodeParamCount, const NDLPass pass) = 0;

    // FindSymbol - Search the engines symbol table for a fully quantified symbol
    // symbol - name of the symbol
    // returns - pointer to the matching EvalValue for that node, of NULL if not found
    virtual void* FindSymbol(const wstring& /*symbol*/)
    {
        return NULL;
    }
    // ProcessOptionalParameters - Process the optional parameters of a node
    // node to process
    virtual void ProcessOptionalParameters(NDLNode<ElemType>* /*node*/)
    {
        return;
    }
};

template class NDLNodeEvaluator<float>;
template class NDLNodeEvaluator<double>;

template <typename ElemType>
class NetNdl // class to associate a network with an NDLScript
{
public:
    ComputationNetworkPtr cn;
    NDLScript<ElemType>* ndl;                // NDLScript we are using for this network. NOTE: the actual script used
    NDLNode<ElemType>* lastNode[ndlPassMax]; // last node we evaluated for each pass
    NetNdl()
        : cn(nullptr), ndl(nullptr)
    {
        ClearLastNodes();
    }
    NetNdl(ComputationNetworkPtr p_cn)
        : cn(p_cn), ndl(nullptr)
    {
        ClearLastNodes();
    }
    NetNdl(ComputationNetworkPtr p_cn, NDLScript<ElemType>* p_ndl)
        : cn(p_cn), ndl(p_ndl)
    {
        ClearLastNodes();
    }
    ~NetNdl()
    {
    }

    // ClearLastNodes - Clear out the last node values for all passes
    void ClearLastNodes()
    {
        for (NDLPass pass = ndlPassInitial; pass < ndlPassMax; ++pass)
        {
            lastNode[pass] = nullptr;
        }
    }

    // Clear - clear out everything in the structure
    // NOTE: this deletes the network and the NDLScript, use with care!
    void Clear()
    {
        cn.reset();
        delete ndl;
        ndl = nullptr;
        ClearLastNodes();
    }
};

template <typename ElemType>
inline NDLNodeEvaluator<ElemType>::~NDLNodeEvaluator()
{
} // defined even though it's virtual; supposed to be faster this way

// NDLNode - Network Description Language Node
// Used to represent a named entity in the NDL
// if a name is not provided (such as in nesting scenarios) one will be generated
template <typename ElemType>
class NDLNode
{
private:
    std::string m_name;            // value on the left of the equals
    ConfigValue m_value;           // value on the right of the equals (CN node name, or value)
    NDLScript<ElemType>* m_parent; // parent script
    NDLType m_type;                // type of node
    ConfigArray m_paramString;     // parameter of a function/array
    ConfigArray m_paramMacro;      // parameter of a macro (the variables used in the macro definition)
    vector<NDLNode*> m_parameters; // parameters as nodes/array elements
    void* m_eval;                  // pointer to an arbitrary eval structure
    NDLScript<ElemType>* m_script; // script for ndlTypeMacro
    static int s_nameCounter;      // counter for generating unique names
public:
    NDLNode(const std::string& name, ConfigValue value, NDLScript<ElemType>* parent, NDLType ndlType)
    {
        if (name.empty())
            GenerateName();
        else
            m_name = name;
        m_value = value;
        m_parent = parent;
        assert(parent != NULL);
        parent->AddChild(this);
        m_type = ndlType;
        m_eval = NULL;
        m_script = NULL;
    }

    ~NDLNode()
    {
    }

    // publicly accessible Copy method
    // should only be used for macro expansion
    NDLNode* Copy() const
    {
        NDLNode* ret = new NDLNode(*this);
        return ret;
    }

private:
    // copy constructor, creates a new disconnected copy of this node for macro expansion
    NDLNode(const NDLNode& copyMe);

    NDLNode& operator=(NDLNode& /*copyMe*/) // this is just a place holder implementation which is not functioning but prevent callers to use it.
    {
        LogicError("'NDLNode& operator=(NDLNode& copyMe)' should never be called.");
    }

    // generate a generic symbol name for a node
    void GenerateName()
    {
        char buffer[10];
        sprintf(buffer, "%d", ++s_nameCounter);
        m_name = std::string("unnamed") + buffer;
    }

public:
    void SetScript(NDLScript<ElemType>* script)
    {
        m_script = script;
    }
    NDLScript<ElemType>* GetScript() const
    {
        return m_script;
    }
    void SetType(NDLType type)
    {
        m_type = type;
    }
    NDLType GetType() const
    {
        return m_type;
    }
    const std::string& GetName() const
    {
        return m_name;
    }
    void SetName(std::string& name)
    {
        m_name = name;
    }
    ConfigValue GetValue() const
    {
        return m_value;
    }
    void SetValue(std::string& value)
    {
        m_value = value;
    }

    // parameters of a function (ndlTypFunction), or parameters in the call to a macro
    void SetParamString(ConfigValue paramString)
    {
        m_paramString = paramString;
    }
    ConfigArray GetParamString() const
    {
        return m_paramString;
    }

    // parameters of a macro
    void SetParamMacro(ConfigValue paramMacro)
    {
        m_paramMacro = paramMacro;
    }
    ConfigArray GetParamMacro() const
    {
        return m_paramMacro;
    }

    void SetParentScript(NDLScript<ElemType>* script)
    {
        m_parent = script;
    }
    NDLScript<ElemType>* GetParentScript()
    {
        return m_parent;
    }

    // get parameters, either just optional or just regular
    vector<NDLNode*> GetParameters(bool optional = false) const
    {
        vector<NDLNode*> result;
        for (NDLNode* param : m_parameters)
        {
            bool optParam = param->GetType() == ndlTypeOptionalParameter;
            if (optParam == optional)
                result.push_back(param);
        }
        return result;
    }

    // Get/Set eval values
    void* GetEvalValue() const
    {
        return m_eval;
    }
    void SetEvalValue(void* evalValue)
    {
        m_eval = evalValue;
    }

    // GetOptionalParameter - Get an optional parameter value
    // name - the name to search for in the optional parameters
    // deflt - the default value (if not found)
    // returns: parameter value if found, or default value otherwise
    ConfigValue GetOptionalParameter(const std::string& name, const std::string& deflt) const
    {
        for (NDLNode* param : m_parameters)
        {
            bool optParam = param->GetType() == ndlTypeOptionalParameter;
            if (optParam && EqualCI(param->GetName(), name))
            {
                auto paramValue = param->GetValue();
                auto resolveParamNode = m_parent->ParseVariable(paramValue, false);
                if (resolveParamNode != nullptr)
                    return resolveParamNode->GetScalar();
                else
                    return paramValue;
            }
        }
        return ConfigValue(deflt);
    }

    // FindNode - Find a node of the given name
    // name - name to search for
    // searchForDotNames - search for NDL symbols traversing call heirarchy
    // returns: The node with that name, or NULL if not found
    NDLNode* FindNode(const std::string& name, bool searchForDotNames = false)
    {
        NDLNode* found = m_parent->FindSymbol(name, searchForDotNames);
        if (!found)
            found = NDLScript<ElemType>::GlobalScript().FindSymbol(name, searchForDotNames);
        return found;
    }

    // GetScalar - Get a scalar value from a node, may loop through some variables before arriving
    // returns: scalar value
    ConfigValue GetScalar()
    {
        NDLNode<ElemType>* node = this;
        while (node && (node->GetType() == ndlTypeVariable || node->GetType() == ndlTypeParameter))
        {
            NDLNode<ElemType>* nodeLast = node;
            node = node->FindNode(node->GetValue(), true /*searchForDotNames*/);

            // if we are still on the same node, that means it was never resolved to anything, an undefined variable
            if (nodeLast == node)
            {
                RuntimeError("undefined Variable, '%s' found, must be declared before first use\n", node->GetName().c_str());
            }
        }
        if (!node || node->GetType() != ndlTypeConstant)
        {
            std::string name = node ? node->GetName() : GetName();
            RuntimeError("Scalar expected, '%s' must be a constant or variable that resolves to a constant\n", name.c_str());
        }
        return node->GetValue();
    }

    void InsertParam(NDLNode* param)
    {
        m_parameters.push_back(param);
    }

    // EvaluateMacro - Evaluate a macro, make the call
    // nodeEval - the node evaluator we are using to interpret the script
    // baseName - base name for all symbols at this level
    // pass - what NDLPass are we in?
    // returns: the return node for this macro
    NDLNode<ElemType>* EvaluateMacro(NDLNodeEvaluator<ElemType>& nodeEval, const wstring& baseName, const NDLPass pass)
    {
        if (m_type != ndlTypeMacroCall)
            return NULL;

        // make sure the actual parameters and expected parameters match
        if (m_parameters.size() < m_paramMacro.size())
        {
            RuntimeError("Parameter mismatch, %d parameters provided, %d expected in call to %s\n",
                         (int) m_parameters.size(), (int) m_paramMacro.size(), m_value.c_str());
        }

        // assign the actual parameters in the script so we can execute it
        for (int i = 0; i < m_parameters.size(); ++i)
        {
            NDLNode<ElemType>* nodeParam = m_parameters[i];
            std::string paramName = i < m_paramMacro.size() ? m_paramMacro[i] : nodeParam->GetName();

            // if the node is a parameter then look it up in the symbol table
            if (nodeParam->GetType() == ndlTypeParameter)
            {
                nodeParam = m_parent->FindSymbol(nodeParam->GetName());
            }
            // do we want to add optional parameters as symbols, or not?
            else if (nodeParam->GetType() == ndlTypeOptionalParameter)
            {
                if (i < m_paramMacro.size())
                    RuntimeError("Parameter mismatch, parameter %d is an optional parameter, but should be a required parameter\n", i);
                // if no symbol yet, add it
                if (!m_script->ExistsSymbol(paramName))
                {
                    m_script->AddSymbol(paramName, nodeParam);
                    continue;
                }
                // else assign the value below
            }

            // assign the parameter symbols in the script we will call with the values passed to the call
            m_script->AssignSymbol(paramName, nodeParam);
        }

        std::wstring newBase = baseName;
        if (!newBase.empty())
            newBase += L".";
        newBase += Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(m_name);

        // now evaluate  the contained macro script
        NDLNode<ElemType>* nodeResult = m_script->Evaluate(nodeEval, newBase, pass);
        // Consider: do we need to restore the original mapping here, may need to for recursive calls?

        // look for a symbol that is identical to the macro name, if it exists this is the return value
        NDLNode<ElemType>* nodeMacroName = m_script->FindSymbol(m_value);
        if (nodeMacroName)
        {
            nodeResult = nodeMacroName;
        }

        // set the eval node to be the same as the return value;
        if (nodeResult)
        {
            m_eval = nodeResult->GetEvalValue();
        }
        return nodeResult;
    }
};

template <typename ElemType>
class NDLScript : public ConfigParser
{
private:
    std::wstring m_baseName;
    std::string m_scriptString;
    std::vector<NDLNode<ElemType>*> m_script;                            // script lines in parsed node order, macros will have definition followed by body
    std::map<std::string, NDLNode<ElemType>*, nocase_compare> m_symbols; // symbol table
    NDLNode<ElemType>* m_macroNode;                                      // set when interpretting a macro definition
    bool m_noDefinitions;                                                // no definitions can be made in this script, interpret all macro/function names as calls
    static NDLScript<ElemType> s_global;                                 // ("global"); // global script for storing macros and global nodes
    std::vector<NDLNode<ElemType>*> m_children;                          // child nodes. Note that m_script nodes may not be children of this object, they include macro nodes
    ComputationNetworkPtr m_cn;                                          // computation network to use for backup symbol lookup. Used for MEL where NDL and network nodes are mixed
    bool m_definingMacro;                                                // currently defining a macro, flag to determine if we are defining or interpretting a macro call

public:
    // constructors that take a config name
    NDLScript(const std::string& configname)
        : ConfigParser(';', configname)
    {
        m_macroNode = NULL;
        m_noDefinitions = false;
        m_definingMacro = false;
    }
    NDLScript(const std::wstring& configname)
        : ConfigParser(';', configname)
    {
        m_macroNode = NULL;
        m_noDefinitions = false;
        m_definingMacro = false;
    }
    ~NDLScript()
    {
        // need to free all the child nodes attached to this script node
        for (NDLNode<ElemType>* node : m_children)
            delete node;
        m_children.clear();
    }

    // empty constructor
    NDLScript()
        : ConfigParser(';')
    {
        m_macroNode = NULL;
        m_noDefinitions = false;
        m_definingMacro = false;
    } // parameterless version if needed

    // construct NDLScript from a ConfigValue, propogate the config Name
    NDLScript(const ConfigValue& configValue)
        : ConfigParser(';', configValue.Name())
    {
        m_macroNode = NULL;
        m_noDefinitions = false;
        m_definingMacro = false;
        m_scriptString = configValue;
        Parse(m_scriptString);
    }

    // construct NDLScript from a ConfigValue, propogate the config Name
    // configValue - the body of the macro
    // oneLineDefinition - this macro definition is all on one line, names optional
    // macroName - if the macro has a name, the name - this is used to get parameter info
    NDLScript(const ConfigValue& configValue, std::string macroName, bool oneLineDefinition)
        : ConfigParser(';', configValue.Name())
    {
        m_noDefinitions = oneLineDefinition;
        m_definingMacro = true;
        m_macroNode = NULL;
        m_scriptString = configValue;
        NDLNode<ElemType>* ndlNode = s_global.CheckName(macroName, true);
        if (ndlNode == NULL)
            RuntimeError("Invalid macro definition, %s not found", macroName.c_str());

        // get and parse the parameters
        ConfigArray parameters = ndlNode->GetParamMacro();
        for (auto iter = parameters.begin(); iter != parameters.end(); ++iter)
        {
            // we are adding parameters that will be replaced by actual values later
            ConfigValue param = *iter;

            // check to make sure this parameter name is not a reserved word
            std::string functionName = param;
            // check for function name, a function may have two valid names
            // in which case 'functionName' will get the default node name returned
            if (CheckFunction(functionName))
            {
                RuntimeError("NDLScript: Macro %s includes a parameter %s, which is also the name of a function. Parameter names may not be the same as function names.", macroName.c_str(), param.c_str());
            }

            NDLNode<ElemType>* paramNode = new NDLNode<ElemType>(param, param, this, ndlTypeParameter);
            // add to node parameters
            ndlNode->InsertParam(paramNode);
            // add to script symbol table
            AddSymbol(param, paramNode);
        }
        Parse(m_scriptString);
        m_definingMacro = false;
    }

    // copy and move constructors
    NDLScript(const NDLScript& copyMe);
    NDLScript(const NDLScript&& moveMe);

private:
    NDLNode<ElemType>* DuplicateNode(NDLNode<ElemType>* node);

public:
    // GlobalScript - Access to global script
    static NDLScript<ElemType>& GlobalScript()
    {
        return s_global;
    }

    // SetMacroDefinitionsAllowed - allow macro definitions
    // macroAllowed - can macros be defined in this script?
    void SetMacroDefinitionsAllowed(bool macroAllowed)
    {
        m_noDefinitions = !macroAllowed;
    }

    void SetBaseName(const std::wstring& baseName)
    {
        m_baseName = baseName;
    }
    const std::wstring& GetBaseName()
    {
        return m_baseName;
    }

    void ClearGlobal()
    {
        s_global.Clear();
    }

    void Clear()
    {

        for (NDLNode<ElemType>* node : m_children)
            delete node;
        m_children.clear();
        for (NDLNode<ElemType>* node : m_script)
            delete node;
        m_script.clear();

        m_symbols.clear();
    }
    void ClearEvalValues()
    {
        for (NDLNode<ElemType>* node : m_children)
        {
            node->SetEvalValue(NULL);
        }
    }
    // AddChild - add a child node to the script
    // node - node to add
    // NOTE: this NDLScript owns this node and is responsible to delete it
    void AddChild(NDLNode<ElemType>* node)
    {
        m_children.push_back(node);
    }

    // SetComputationNetwork - set the computation network this NDL is associated with
    void SetComputationNetwork(ComputationNetworkPtr cn)
    {
        m_cn = cn;
    }

    // FindSymbol - Find a symbol to the symbol table
    // symbol - symbol to find
    // searchForDotNames - search for NDL symbols traversing call heirarchy
    // returns - node this symbol references
    NDLNode<ElemType>* FindSymbol(const std::string& symbol, bool searchForDotNames = true)
    {
        auto found = m_symbols.find(symbol); // search symbol directly first
        if (found != m_symbols.end())
            return found->second;

        // if not found, handle dot names by move up the hierarchy
        size_t firstDot = symbol.find_first_of('.');
        if (firstDot == npos)
            return nullptr;

        std::string search = symbol.substr(0, firstDot);
        found = m_symbols.find(search);
        if (found == m_symbols.end())
        {
            return NULL;
        }

        // handle dot names,
        if (firstDot != npos)
        {
            NDLNode<ElemType>* node = found->second;
            NDLScript<ElemType>* script = node->GetScript();
            // if there is no script, probably a parameter/variable with further 'dot' values (ie. var.CE.BFF)
            if (script != NULL)
            {
                if (node->GetType() != ndlTypeMacroCall || script == NULL)
                    RuntimeError("Symbol name not valid, %s is not a macro, so %s cannot be interpretted", search.c_str(), symbol.c_str());
                return script->FindSymbol(symbol.substr(firstDot + 1), searchForDotNames);
            }
        }
        return found->second;
    }

    // ExistsSymbol - Find if a symbol exists (value might be NULL)
    // symbol - symbol to find
    // returns - true if it's there
    bool ExistsSymbol(const std::string& symbol)
    {
        auto found = m_symbols.find(symbol);
        return (found != m_symbols.end());
    }

    // ContainsOptionalParameter - do any nodes in this script have an optional parameter by the following name?
    // optParamName - name of parameter we are searching for
    // returns: vector of the nodes found (empty if nothing found)
    vector<NDLNode<ElemType>*> ContainsOptionalParameter(const std::string& optParamName)
    {
        vector<NDLNode<ElemType>*> result;
        std::string empty;
        for (auto& symbol : m_symbols)
        {
            NDLNode<ElemType>* node = symbol.second;
            std::string value = node->GetOptionalParameter(optParamName, empty);
            if (!value.empty())
            {
                result.push_back(node);
            }
        }
        return result;
    }

    // AddSymbol - Add a symbol to the symbol table
    // symbol - symbol to add
    // node - node this symbol references
    // NOTE: at present we don't allow reuse of a symbol, so this throws an error if it sees an existing symbol
    void AddSymbol(const std::string& symbol, NDLNode<ElemType>* node)
    {
        auto found = m_symbols.find(symbol);
        if (found != m_symbols.end())
        {
            NDLNode<ElemType>* nodeFound = found->second;
            // check for undetermined nodes, because these nodes are to be defined later
            if (nodeFound->GetType() != ndlTypeUndetermined && nodeFound->GetType() != ndlTypeParameter)
            {
                std::string value = found->second->GetValue();
                RuntimeError("Symbol '%s' currently assigned to '%s' reassigning to a different value not allowed\n", symbol.c_str(), value.c_str());
            }
        }
        m_symbols[symbol] = node;
    }

    // AssignSymbol - Assign a new value to a symbol in the table
    // symbol - symbol to assign
    // node - node this symbol will reference
    void AssignSymbol(const std::string& symbol, NDLNode<ElemType>* node)
    {
        auto found = m_symbols.find(symbol);
        if (found == m_symbols.end())
        {
            RuntimeError("Symbol '%s' currently does not exist, attempting to assigned value '%s' AssignSymbol() requires existing symbol\n", symbol.c_str(), node->GetValue().c_str());
        }
        m_symbols[symbol] = node;
    }

    // FileParse - parse at the file level, can be overridden for "section of file" behavior
    // stringParse - file concatentated as a single string
    void FileParse(const std::string& stringParse)
    {
        ConfigParameters sections(stringParse);
        bool loadOrRunFound = false;

        // load all the sections that we want (macros)
        if (sections.Exists("load"))
        {
            auto config = ConfigArray(sections("load"));
            for (int i = 0; i < config.size(); ++i)
            {
                Parse(sections(config[i]));
            }
            loadOrRunFound = true;
        }

        // load and then execute
        if (sections.Exists("run"))
        {
            auto config = ConfigArray(sections("run"));
            for (int i = 0; i < config.size(); ++i)
            {
                Parse(sections(config[i]));
            }
            loadOrRunFound = true;
        }

        // didn't find any of the tags, so just parse the whole thing as a script
        if (!loadOrRunFound)
        {
            // surround text in braces so we parse correctly
            std::string textInBraces = "[ " + stringParse + " ]";
            Parse(textInBraces);
        }
    }

    // IsMacroDefinition - is this a macro definition?
    // returns - true if a definition, otherwise false
    bool IsMacroDefinition()
    {
        return m_definingMacro;
    }

    // CheckName - check for a name in our symbols, see if it exists
    // name - name we are looking for
    // localOnly - only look in the current scope, and not the global scope
    // if it does exist return the node that represents the name
    NDLNode<ElemType>* CheckName(const std::string& name, bool localOnly = false)
    {
        // first try local script
        auto found = FindSymbol(name);
        if (found != NULL)
        {
            return found;
        }

        // next try the globals, this includes macros and global constants
        if (!localOnly)
        {
            auto found2 = s_global.FindSymbol(name);
            if (found2 != NULL)
            {
                NDLNode<ElemType>* node = found2;
                if (node->GetType() == ndlTypeMacro)
                {
                    // if we are calling a macro we need to keep track of formal parameters,
                    // keep them as strings in this macroCall node
                    NDLNode<ElemType>* newNode = new NDLNode<ElemType>("", name, this, ndlTypeMacroCall);
                    NDLScript<ElemType>* script = node->GetScript();

                    // if this is a macro call (and not a definition), we want to expand the macro (make a copy)
                    if (!IsMacroDefinition())
                    {
                        script = new NDLScript<ElemType>(*script);
                    }
                    newNode->SetScript(script);

                    newNode->SetParamMacro(node->GetParamMacro());
                    node = newNode;
                }
                return node;
            }
        }

        std::string functionName = name;
        // check for function name, a function may have two valid names
        // in which case 'functionName' will get the default node name returned
        if (CheckFunction(functionName))
        {
            NDLNode<ElemType>* ndlNode = new NDLNode<ElemType>("", functionName, this, ndlTypeFunction);
            return ndlNode;
        }

        // not found, return NULL
        return NULL;
    }

    // CallStringParse - parse the string description of a call sequence
    // token - [in] string description of the call
    // nameFunction - [out] name of the function being called
    // params - [out] parameters to the function, set to empty string if no parameters
    // returns: the node (if it exists) that matches this function name, otherwise NULL
    NDLNode<ElemType>* CallStringParse(const std::string& token, std::string& nameFunction, std::string& params)
    {
        auto paramStart = token.find_first_of(OPENBRACES);
        if (paramStart == npos)
            RuntimeError("Invalid macro/function call can not be parsed: %s\n", token.c_str());
        nameFunction = token.substr(0, paramStart);
        Trim(nameFunction);
        params = token.substr(paramStart);
        NDLNode<ElemType>* ndlNodeFound = CheckName(nameFunction);
        return ndlNodeFound;
    }

    // ParseParameters - parse the parameters of a macro, or an array
    // ndlNode - node we should add the parameters to
    // value - parameters as config value
    // createNew - create a new parameter node if one does not exist
    void ParseParameters(NDLNode<ElemType>* ndlNode, const ConfigValue& value, bool createNew = false)
    {
        ConfigArray parameters = value;
        for (auto iter = parameters.begin(); iter != parameters.end(); ++iter)
        {
            ConfigValue param = *iter;
            NDLNode<ElemType>* paramNode = NULL;
            auto foundBrace = param.find_first_of(FUNCTIONOPEN);
            if (foundBrace != npos) // a nested call as a parameter
                paramNode = ParseCall(param);
            else // must be predefined variable or constant
            {
                paramNode = ParseVariable(param, createNew);

                // if we can't find the node right now, it's undetermined, must be defined later, or throw an error later
                if (paramNode == nullptr)
                {
                    paramNode = new NDLNode<ElemType>(param, param, this, ndlTypeUndetermined);
                    // add to the symbol table
                    AddSymbol(param, paramNode);
                }
            }
            if (paramNode == NULL)
            {
                RuntimeError("variable name '%s' not found, must be previously defined\n", param.c_str());
            }
            else
            {
                ndlNode->InsertParam(paramNode);
            }
        }
    }

    // ParseVariable - parse a variable or constant
    // token - string containing the variable or constant
    // createNew - create a new variable node if no node found
    // returns: the node that represents this newly defined variable
    NDLNode<ElemType>* ParseVariable(const std::string& token, bool createNew = true)
    {
        NDLNode<ElemType>* ndlNode = NULL;
        auto openBrace = token.find_first_of(OPENBRACES);
        if (openBrace == 0)
        {
            ndlNode = new NDLNode<ElemType>("", token, this, ndlTypeArray);
            ndlNode->SetParamString(token);
            ParseParameters(ndlNode, token);
            return ndlNode;
        }

        char* pEnd;
        strtod(token.c_str(), &pEnd);

        // see if it's a numeric constant
        if (*pEnd == 0)
        {
            ndlNode = new NDLNode<ElemType>("", token, this, ndlTypeConstant);
        }
        // not a constant, so must be a variable
        else
        {
            // look for an optional parameter
            auto foundEqual = token.find_first_of('=');
            bool optional = (foundEqual != npos);
            if (optional)
            {
                std::string name = token.substr(0, foundEqual);
                Trim(name);
                std::string value = token.substr(foundEqual + 1);
                Trim(value);
                TrimQuotes(value); // strip enclosing quotes

                ndlNode = new NDLNode<ElemType>(name, value, this, ndlTypeOptionalParameter);
            }
            else
            {
                ndlNode = CheckName(token);
                if (createNew && ndlNode == NULL)
                {
                    // NOTE: currently we only get here in Parameter scenarios,
                    // if other scenarios present themselves, need a good way to change the type
                    ndlNode = new NDLNode<ElemType>(token, token, this, ndlTypeParameter);
                    AddSymbol(token, ndlNode);
                }
            }
        }
        return ndlNode;
    }

    // ParseDefinition - parse a macro definition
    // token - string containing the macro definition (without the macro body)
    // returns: the node that represents this newly defined macro
    NDLNode<ElemType>* ParseDefinition(const std::string& token)
    {
        std::string nameFunction, params;
        NDLNode<ElemType>* ndlNode = CallStringParse(token, nameFunction, params);
        if (ndlNode)
            RuntimeError("function '%s' already defined\n", nameFunction.c_str());
        ndlNode = new NDLNode<ElemType>(nameFunction, params, &s_global, ndlTypeMacro);

        // now set the variables/parameters which will be parsed when the body shows up
        ndlNode->SetParamMacro(params);

        // now add this to the globals
        s_global.AddSymbol(nameFunction, ndlNode);

        // NOTE: the body of the Macro will be parsed separately, this just sets up the node
        return ndlNode;
    }

    // ParseCall - parse the call syntax out into "function" and variables
    // token - string containing the "call"
    // return - Node pointer, the newly created node
    NDLNode<ElemType>* ParseCall(const std::string& token)
    {
        std::string nameFunction, params;
        NDLNode<ElemType>* ndlNode = CallStringParse(token, nameFunction, params);

        if (ndlNode == NULL)
            RuntimeError("Undefined function or macro '%s' in %s\n", nameFunction.c_str(), token.c_str());

        // now setup the variables/parameters
        ConfigValue value = ConfigValue(params, nameFunction);

        ndlNode->SetParamString(value);
        ParseParameters(ndlNode, value);
        return ndlNode;
    }

    // parse a 'key=value' pair and create the appropriate node for what was seen
    // 'key=Function(x,y,z)' - function
    // 'macro(x,y)={z=Input(x,y)}
    // may also be Function(x,y,z), a nameless call (used in one-line macros)
    std::string::size_type ParseValue(const std::string& stringParse, std::string::size_type tokenStart, std::string::size_type tokenEnd)
    {
        // first find previous character

        // skip leading spaces
        tokenStart = stringParse.find_first_not_of(" \t", tokenStart);
        auto keyEnd = stringParse.find_first_of(OPENBRACES "=", tokenStart);
        bool equalFound = (keyEnd != npos && keyEnd < tokenEnd && stringParse[keyEnd] == '=');

        // this should be the body of the macro
        if (m_macroNode)
        {
            bool oneLineDefinition = false;
            NDLNode<ElemType>* macroNode = m_macroNode;

            // an '=' at the beginning, skip it
            if (keyEnd == tokenStart && equalFound)
            {
                // skip the '=' sign
                oneLineDefinition = true;
                tokenStart = stringParse.find_first_not_of(" \t", tokenStart + 1);
                if (tokenStart == npos)
                    RuntimeError("Body of Macro missing");
            }

            NDLScript<ElemType>* script = new NDLScript<ElemType>(ConfigValue(stringParse.substr(tokenStart, tokenEnd - tokenStart), macroNode->GetName()), macroNode->GetName(), oneLineDefinition);
            macroNode->SetScript(script);

            // reset so we know we are done with the body
            m_macroNode = NULL;

            return tokenEnd; // done with the macro now
        }

        // if we hit the end of the token before we hit an equal sign, it's a 'macro(x,y)' definition
        // unless we are a one-line macro in which case we don't allow definitions
        if (!m_noDefinitions && !equalFound)
        {
            keyEnd = stringParse.find_first_of(OPENBRACES, tokenStart);
            if (keyEnd == npos || keyEnd >= tokenEnd)
                RuntimeError("Invalid statement, does not contain an '=' sign: %s\n", stringParse.substr(tokenStart, tokenEnd - tokenStart).c_str());
            m_macroNode = ParseDefinition(stringParse.substr(tokenStart, tokenEnd - tokenStart));
            // the body of the macro will come through next time
            return tokenEnd;
        }

        // get the key value (symbol name)
        std::string key;

        // no macro definitions allowed, so no equal means a function call
        if (m_noDefinitions && !equalFound)
        {
            ; // nothing to do here, just skip the "key=" parsing below
        }
        else
        {
            key = stringParse.substr(tokenStart, keyEnd - tokenStart);
            Trim(key);

            // check to make sure variable name isn't a valid function name as well
            string strTemp = key;
            if (CheckFunction(strTemp))
                RuntimeError("variable %s is invalid, it is reserved because it is also the name of a function", key.c_str());

            tokenStart = keyEnd;
            if (stringParse[keyEnd] == '=')
                ++tokenStart;

            // skip any spaces before the second token
            tokenStart = stringParse.find_first_not_of(" \t", tokenStart);
        }
        std::string::size_type substrSize = tokenEnd - tokenStart;

        auto bracesEnd = FindBraces(stringParse, tokenStart);

        // if braces found, we modify the token end according to braces
        if (bracesEnd != npos)
        { // include the trailing brace
            tokenEnd = bracesEnd + 1;
            substrSize = tokenEnd - tokenStart;

            // for quote delimited string remove quotes
            if (stringParse[tokenStart] == '"')
            {
                tokenStart++;
                substrSize -= 2; // take out the quotes
            }
        }

        if (substrSize == 0)
            return npos;

        // get the value
        std::string value = stringParse.substr(tokenStart, substrSize);
        Trim(value);

        NDLNode<ElemType>* ndlNode = NULL;

        // check for a function/macro call
        auto found = value.find_first_of(FUNCTIONOPEN);
        if (found != npos && found > 0) // brace found after some text, so a call
        {
            ndlNode = ParseCall(value);
            // check if we have a user defined name, ParseCall assigns a default name
            if (!key.empty())
                ndlNode->SetName(key);
            AddSymbol(ndlNode->GetName(), ndlNode);
            m_script.push_back(ndlNode);
        }
        // if it's not a call, must be a variable
        else
        {
            ndlNode = ParseVariable(value);
            bool newNode = ndlNode->GetName().empty();
            AddSymbol(key, ndlNode);

            ndlNode->SetName(key);
            if (newNode) // only need to add nodes that are new (not renames)
            {
                m_script.push_back(ndlNode);
            }
        }

        return tokenEnd;
    }

    // ExpandMacro - Expand a macro into a new macro definition
    // node - NDLNode that holds the macro call
    // returns: new node with the expanded macro
    NDLNode<ElemType>* ExpandMacro(const NDLNode<ElemType>* node)
    {
        assert(node->GetType() == ndlTypeMacroCall); // needs to be a macro call (not definition)

        std::string name = node->GetName();
        // if we are calling a macro make a new copy of it and execute that instead (macro expansion)
        // we do this so the evalValues in the macros will be valid regardless of number of instantiations
        NDLNode<ElemType>* newNode = new NDLNode<ElemType>(name, node->GetValue(), this, ndlTypeMacroCall);
        NDLScript<ElemType>* newScript = new NDLScript<ElemType>(*node->GetScript());
        newNode->SetScript(newScript);
        newNode->SetParamMacro(node->GetParamMacro());

        // now get the parameters to the macro added
        ConfigValue paramString = node->GetParamString();
        ParseParameters(newNode, paramString, true /*createNew*/);
        newNode->SetParamString(paramString);

        // fixup the symbol table to point to this one instead
        AssignSymbol(name, newNode);
        return newNode;
    }

    // Evaluate - Evaluate the script
    // nodeEval - the node evaluator to call
    // baseName - baseName for all labels
    // pass - what NDLPass are we on?
    // skipThrough - skip through this node, will skip eval for all nodes up to and including this one
    NDLNode<ElemType>* Evaluate(NDLNodeEvaluator<ElemType>& nodeEval, const wstring& baseName, const NDLPass pass = ndlPassInitial, NDLNode<ElemType>* skipThrough = nullptr)
    {
        NDLNode<ElemType>* nodeLast = skipThrough;
        bool skip = skipThrough != nullptr;
        std::wstring prevBaseName = GetBaseName();
        SetBaseName(baseName);

        for (auto& node : m_script)
        {
            // if we are in skip mode, and we found the skipThrough node,
            // move out of skip mode and start processing at next node
            if (skip)
            {
                if (node == skipThrough)
                    skip = false;
                continue;
            }

            // if it's a macro call, call the macro
            if (node->GetType() == ndlTypeMacroCall)
            {
                node->EvaluateMacro(nodeEval, baseName, pass);
                nodeEval.ProcessOptionalParameters(node);
            }
            else
            {
                nodeEval.Evaluate(node, baseName, pass);
            }
            nodeLast = node;
        }
        SetBaseName(prevBaseName);
        return nodeLast;
    }
};
} } }
