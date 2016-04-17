//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "Config.h"
#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"
#include "NetworkDescriptionLanguage.h"
#include "NDLNetworkBuilder.h"
#include "NDLUtil.h"
#include <stdarg.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// EqualInsensitive - check to see if two nodes are equal up to the length of the first string (must be at least half as long as actual node name)
// string1 - [in,out] string to compare, if comparision is equal insensitive but not sensitive, will replace with sensitive version
// string2 - second string to compare
// alternate - alternate naming of the string
// return - true if strings are equal insensitive and modifies string1 to sensitive version if different
bool EqualInsensitive(std::string& string1, const char* string2, const char* alternate = NULL);

template <typename ElemType>
class MELScript : public ConfigParser
{
private:
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    typedef std::map<std::string, NetNdl<ElemType>, nocase_compare> MapNameToNetNdl;

    std::string m_scriptString;
    MapNameToNetNdl m_mapNameToNetNdl; // computational networks
    NetNdl<ElemType>* m_netNdlDefault;
    NDLScript<ElemType> m_ndlScript; // one to store the macros

public:
    typedef map<ComputationNodePtr, ComputationNodePtr> MapNodes;

    // constructors that take a config name
    MELScript(const std::string& configname)
        : ConfigParser(';', configname)
    {
    }
    MELScript(const std::wstring& configname)
        : ConfigParser(';', configname)
    {
    }

    // cleanup all the computational networks we loaded
    ~MELScript()
    {
        for (auto iter : m_mapNameToNetNdl)
        {
            iter.second.Clear();
        }
        m_mapNameToNetNdl.clear();
    }

    // empty constructor
    MELScript()
        : ConfigParser(';')
    {
    } // parameterless version if needed

    // construct MELScript from a ConfigValue, propogate the config Name
    MELScript(const ConfigValue& configValue)
        : ConfigParser(';', configValue.Name())
    {
        m_scriptString = configValue;
        Parse(m_scriptString);
    }

    // copy and move constructors
    MELScript(const MELScript& melScript)
        : ConfigParser(melScript)
    {
        m_scriptString = melScript.m_scriptString;
        m_mapNameToNetNdl = melScript.m_mapNameToNetNdl; // computational networks
        m_netNdlDefault = melScript.m_netNdlDefault;
        // don't need to copy m_ndlScript, only used to store macros (which are stored in global instance anyway)
    }
    MELScript(const MELScript&& melScript)
        : ConfigParser(move(melScript))
    {
        m_scriptString = move(melScript.m_scriptString);
        m_mapNameToNetNdl = move(melScript.m_mapNameToNetNdl); // computational networks
        m_netNdlDefault = move(melScript.m_netNdlDefault);
    }
    void ProcessNDLScript(NetNdl<ElemType>* netNdl, NDLPass ndlPassUntil = ndlPassAll, bool fullValidate = false);
    void SetGroupTag(ComputationNodeBasePtr nodeProp, ComputationNetworkPtr cn, const std::wstring& groupTag, bool set);
    void CallFunction(const std::string& name, const ConfigParamList& params);

    // ParseName - Parse the name and find positions of the wildcard matches
    // name - name to parse
    // firstStart - [out] starting position of the first section
    // firstCount - [out] length of the first section
    // secondStart - [out] starting position of the second section
    // secondCount - [out] length of the second section
    // returns: NetNdl structure that matches the prefix name
    // Notes: This functions handles the following patterns:
    // (prefix, prefix*, prefix.*, prefix.first, first, prefix.first*, first*, first*second, prefix.first*second)
    // prefix names must not be identical to a "first" name or it will be distinguished as a prefix
    NetNdl<ElemType>* ParseName(std::string name, size_t& firstStart, size_t& firstCount, size_t& secondStart, size_t& secondCount)
    {
        size_t firstDot = name.find_first_of(".*");
        NetNdl<ElemType>* netNdl = m_netNdlDefault;

        // find the where the first section starts (if there is one)
        std::string symb = name.substr(0, firstDot);
        auto found = m_mapNameToNetNdl.find(symb);
        if (found != m_mapNameToNetNdl.end())
        {
            firstStart = (firstDot == npos ? name.length() : firstDot + 1);
            netNdl = &found->second;
        }
        else
        {
            firstStart = 0;
        }

        // determine where the second element starts (after the asterisk)
        size_t foundStar = name.find_first_of('*', firstStart);
        if (foundStar != npos)
        {
            firstCount = foundStar - firstStart;
            secondStart = foundStar + 1;
            secondCount = name.length() - secondStart;
        }
        else
        {
            firstCount = name.length() - firstStart;
            secondCount = 0;
            secondStart = name.length();
        }

        return netNdl;
    }

    // FindSymbol - Find matching symbols in the symbol table
    // symbol - symbol to find
    // netNdl - [out] netNdl associated with this symbol
    // returns - nodes this symbol references, might be empty
    vector<ComputationNodeBasePtr> FindSymbols(const std::string& symbol, NetNdl<ElemType>*& netNdl)
    {
        size_t firstStart, firstCount, secondStart, secondCount;
        netNdl = ParseName(symbol, firstStart, firstCount, secondStart, secondCount);
        // take off the prefix
        std::string search;
        if (firstStart == symbol.length())
        {
            // this case is just the model label with nothing else, in that case we want the all nodes
            search = "*";
        }
        else
        {
            search = symbol.substr(firstStart);
        }

        ComputationNetworkPtr cn = netNdl->cn;
        wstring name = msra::strfun::utf16(search);
        vector<ComputationNodeBasePtr> nodes = cn->GetNodesFromName(name);
        // didn't find the name in the current symbols, try NDL
        if (nodes.empty() && netNdl->ndl != nullptr)
        {
            ComputationNetworkBuilder<ElemType> builder(*cn);
            NDLNode<ElemType>* ndlNode = netNdl->ndl->FindSymbol(search);
            if (ndlNode != nullptr)
            {
                ComputationNodePtr value = ComputationNode<ElemType>::FromVoidPtr(ndlNode->GetEvalValue());
                if (value != nullptr)
                {
                    nodes.push_back(ComputationNode<ElemType>::FromVoidPtr(ndlNode->GetEvalValue()));
                }
                else
                {
                    if (ndlNode->GetType() != ndlTypeConstant)
                        RuntimeError("Matching NDL name found for %s, but no corresponding computation node found\n", symbol.c_str());
                    // probably a constant node, so make the ComputationNode that is equivalent
                    auto nodePtr = builder.CreateLearnableParameter(name, 1, 1);
                    ndlNode->SetEvalValue(nodePtr.get());
                    ElemType val = ndlNode->GetScalar();
                    nodePtr->Value().SetValue(val);
                }
            }
        }
        if (nodes.empty())
            RuntimeError("FindSymbols could not find a symbol for %s\n", symbol.c_str());
        return nodes;
    }

    // GenerateNames - Generates a mapping table from original node to destination name
    //    used for wildcard matches (i.e. L1* = L2*)
    // symbolIn - symbol(s) to copy from
    // symbolOut - symbol(s) to copy to
    // netNdlIn -  netNdl to copy from
    // netNdlOut - netNdl to copy to
    // returns - Source nodes and Target names
    typedef pair<ComputationNodeBasePtr, std::wstring> GenNameValue;
    vector<GenNameValue> GenerateNames(const std::string& symbolIn, const std::string& symbolOut, NetNdl<ElemType>*& netNdlIn, NetNdl<ElemType>*& netNdlOut)
    {
        MapNodes mapInOut;
        size_t firstStart, firstCount, secondStart, secondCount;
        netNdlIn = ParseName(symbolIn, firstStart, firstCount, secondStart, secondCount);

        bool inWildcard = !(firstStart + firstCount == secondStart && secondCount == 0);

        // take off the prefix
        std::string search;
        if (firstStart == symbolIn.length())
        {
            // this case is just the model label with nothing else, in that case we want the all nodes
            search = "*";
            inWildcard = true;
        }
        else
        {
            search = symbolIn.substr(firstStart);
        }

        wstring name = msra::strfun::utf16(search);
        vector<ComputationNodeBasePtr> nodes = netNdlIn->cn->GetNodesFromName(name);

        if (!nodes.size()) // found
            RuntimeError("GenerateNames: Node name does not exist %ls.", name.c_str());

        size_t firstStartOut, firstCountOut, secondStartOut, secondCountOut;
        netNdlOut = ParseName(symbolOut, firstStartOut, firstCountOut, secondStartOut, secondCountOut);

        // check for wildcards on input and output
        bool outWildcard = !(firstStartOut + firstCountOut == secondStartOut && secondCountOut == 0);

        // take off the prefix
        if (firstStartOut == symbolOut.length())
        {
            // this case is just the model label with nothing else, in that case we want the all nodes
            search = "*";
            outWildcard = true;
        }
        else
        {
            search = symbolOut.substr(firstStartOut);
        }

        wstring nameOut = msra::strfun::utf16(search);

        bool singleInputMultiOutput = (outWildcard && !inWildcard);

        // valid operations
        // out*=in*, out*second = in*second
        // make sure the patterns are the same
        if (!(singleInputMultiOutput || ((!firstCount == !firstCountOut) && (!secondCount == !secondCountOut))))
        {
            RuntimeError("The input symbols and output symbols must match, when the input matches has more than one element. %s = %s not allowed", symbolOut.c_str(), symbolIn.c_str());
        }

        // get the first and last "unchanged" portions
        std::wstring first = msra::strfun::utf16(symbolOut.substr(firstStartOut, firstCountOut));
        std::wstring second = msra::strfun::utf16(symbolOut.substr(secondStartOut, secondCountOut));

        // now we have the original names from the input symbol, generate the output names
        vector<GenNameValue> ret;

        // if we have more than one matching parameter
        if (singleInputMultiOutput)
        {
            auto nodeIn = nodes[0];
            vector<ComputationNodeBasePtr> nodesOut = netNdlOut->cn->GetNodesFromName(nameOut);

            // make sure that there are some nodes to copy to
            if (nodesOut.size() == 0)
                RuntimeError("Setting a single input to multiple outputs requires the multiple outputs to exist. In %ls = %ls, %ls does not match any nodes.", nameOut.c_str(), name.c_str(), nameOut.c_str());

            // this is the *.W = L2.W case
            // We want to find all the destination existing matches and then assign the in node to all of them
            for (const auto& node : nodesOut)
            {
                std::wstring nodeOutName = node->NodeName();
                GenNameValue value(nodeIn, nodeOutName);
                ret.push_back(value);
            }
        }
        else
        {
            // we are matching up one for one the input to output
            for (auto node : nodes)
            {
                std::wstring nodeName = node->NodeName();
                size_t start = firstCount;
                size_t len = nodeName.length() - start - secondCount;
                assert(start <= nodeName.length());
                assert(len >= 0);
                std::wstring nodeOutName = first + nodeName.substr(start, len) + second;

                GenNameValue value(node, nodeOutName);
                ret.push_back(value);
            }
        }

        // set the return values
        return ret;
    }

    // CopyNodes - Copy nodes from one model to another model
    // symbolIn - symbol(s) to copy from
    // symbolIn - symbol(s) to copy to
    // copyFlags - flags on how to copy the nodes
    void CopyNodes(const std::string& symbolIn, const std::string& symbolOut, CopyNodeFlags copyFlags)
    {
        // get the nodes
        NetNdl<ElemType>* netNdlTo;
        NetNdl<ElemType>* netNdlFrom;
        vector<GenNameValue> copyNodes = GenerateNames(symbolIn, symbolOut, netNdlFrom, netNdlTo);
        map<ComputationNodeBasePtr, ComputationNodeBasePtr> mapCopied; // map from old nodes to new nodes

        // Process any outstanding NDL Scripts
        bool crossNetwork = netNdlTo->cn != netNdlFrom->cn;
        ProcessNDLScript(netNdlFrom, ndlPassAll);
        if (crossNetwork)
        {
            ProcessNDLScript(netNdlTo, ndlPassAll);
        }

        // if we are copying children, let the routine know we are handling cross network children
        if ((copyFlags & CopyNodeFlags::copyNodeInputLinks) && crossNetwork)
            copyFlags = CopyNodeFlags(copyFlags | CopyNodeFlags::copyNodeAcrossNetworks);

        // now we have the original names from the input symbol, generate the output names
        for (GenNameValue name : copyNodes)
        {
            auto& node = name.first;
            std::wstring nodeName = node->NodeName();
            std::wstring nodeOutName = name.second;

            ComputationNodeBasePtr newNode = netNdlTo->cn->CopyNode(*netNdlFrom->cn, nodeName, nodeOutName, copyFlags);
            mapCopied[node] = newNode;
        }

        // if we are doing a children link copy as well, so set the links up if the nodes were copied
        if (copyFlags & CopyNodeFlags::copyNodeInputLinks)
        {
            // loop through the nodes that were copied and fixup all the child links
            for (GenNameValue nodeVal : copyNodes)
            {
                ComputationNodeBasePtr fromNode = nodeVal.first;
                ComputationNodeBasePtr toNode = mapCopied[fromNode];
                for (int i = 0; i < fromNode->GetNumInputs(); i++)
                {
                    auto found = mapCopied.find(fromNode->GetInputs()[i]);
                    auto newNode = (found == mapCopied.end()) ? ComputationNodePtr() : found->second;
                    toNode->SetInput(i, newNode);
                }
            }
        }
    }

    // CopySubTree - Copy subtree from one model to another model
    // symbolIn - symbol(s) to copy from
    // symbolOut - symbol(s) to copy to
    // copyFlags - flags on how to copy the nodes
    void CopySubTree(const std::string& symbolFrom, const std::string& toCNName, const std::string toNamePrefix, CopyNodeFlags copyFlags)
    {
        // get the nodes
        NetNdl<ElemType>* netNdlFrom;

        vector<ComputationNodeBasePtr> fromNodes = FindSymbols(symbolFrom, netNdlFrom);
        size_t firstStart, firstCount, secondStart, secondCount;
        NetNdl<ElemType>* netNdlTo = ParseName(toCNName, firstStart, firstCount, secondStart, secondCount);

        // Process any outstanding NDL Scripts
        bool crossNetwork = netNdlTo->cn != netNdlFrom->cn;
        ProcessNDLScript(netNdlFrom, ndlPassAll);
        if (crossNetwork)
        {
            ProcessNDLScript(netNdlTo, ndlPassAll);
        }

        std::wstring toNamePrefixW = msra::strfun::utf16(toNamePrefix);

        // now we have the original names from the input symbol, generate the output names
        for (int i = 0; i < fromNodes.size(); i++)
        {
            ComputationNodeBasePtr fromNode = fromNodes[i];
            std::wstring fromNodeName = fromNode->NodeName();

            netNdlTo->cn->CopySubTree(*netNdlFrom->cn, fromNodeName, toNamePrefixW, copyFlags);
        }
    }

    void OverrideModelNameAndSetDefaultModel(ComputationNetworkPtr cn, string modelName = "default")
    {
        auto found = m_mapNameToNetNdl.find(modelName);
        if (found != m_mapNameToNetNdl.end() && found->second.cn != cn)
        {
            // clear out the default if needed
            if (m_netNdlDefault == &found->second)
                m_netNdlDefault = nullptr;
            fprintf(stderr, "WARNING: conflicting model name %s. Deleting the old model with the same name.", modelName.c_str());
            found->second.Clear();
        }

        m_mapNameToNetNdl[modelName] = NetNdl<ElemType>(cn); // newly loaded model will be the new default if none has been set yet
        if (m_netNdlDefault == nullptr)
        {
            m_netNdlDefault = &m_mapNameToNetNdl[modelName];
        }
    }

    void SetExistingModelAsDefault(string modelName)
    {
        auto found = m_mapNameToNetNdl.find(modelName);
        if (found == m_mapNameToNetNdl.end())
            RuntimeError("Model %s does not exist. Cannot set it to default.", modelName.c_str());
        else
            m_netNdlDefault = &found->second;
    }

    bool GetOptionalIncludeDataValue(const ConfigParamList& params, const size_t numFixedParams)
    {
        bool includeData = false;
        for (size_t paramNumber = params.size(); paramNumber > numFixedParams; paramNumber--)
        {
            // process optional parameter if it exists
            std::string propName, value;
            if (OptionalParameter(params[paramNumber - 1], propName, value))
            {
                if (EqualInsensitive(propName, "includeData"))
                {
                    includeData = ConfigValue(value);
                }
                else
                {
                    RuntimeError("Invalid optional parameter %s, valid optional parameters: includeData=(false|true)", propName.c_str());
                }
            }
        }

        return includeData;
    }
    wstring GetOptionalModelFormat(const ConfigParamList& params, const size_t numFixedParams)
    {
        wstring modelFormat = L"cntk"; // default
        for (size_t paramNumber = params.size(); paramNumber > numFixedParams; paramNumber--)
        {
            // process optional parameter if it exists
            std::string propName, value;
            if (OptionalParameter(params[paramNumber - 1], propName, value))
            {
                if (EqualInsensitive(propName, "format"))
                {
                    if (EqualInsensitive(value, "cntk"))
                    {
                        modelFormat = L"cntk";
                    }
                    else if (EqualInsensitive(value, "cntk_legacy_no_tensorlib")) // model of late 2015 which had a bug in setting InputValue's tensor dimensions
                    {
                        modelFormat = L"cntk_legacy_no_tensorlib";
                    }
                    else
                    {
                        RuntimeError("Invalid optional parameter value %s, valid values are: format=(cntk)", value.c_str());
                    }
                }
                else
                {
                    RuntimeError("Invalid optional parameter %s, valid optional parameters: format=(cntk)", propName.c_str());
                }
            }
        }

        return modelFormat;
    }

    std::string GetOptionalSnippetSection(const ConfigParamList& params, const size_t numFixedParams)
    {
        // process optional parameter if it exists
        std::string propName, value;
        for (size_t paramNumber = params.size(); paramNumber > numFixedParams; paramNumber--)
        {
            if (OptionalParameter(params[paramNumber - 1], propName, value))
            {
                if (EqualInsensitive(propName, "section"))
                {
                    if (value.empty())
                    {
                        RuntimeError("Invalid optional parameter value <empty>, a section name must be specified: section=(sectionName)");
                    }
                }
                else
                {
                    RuntimeError("Invalid optional parameter %s, valid optional parameters: section=(sectionName)", propName.c_str());
                }
            }
        }

        return value;
    }

    CopyNodeFlags GetOptionalCopyNodeFlags(const ConfigParamList& params, const size_t numFixedParams)
    {
        CopyNodeFlags copyFlags = CopyNodeFlags::copyNodeAll; // by default copy both values and link structure

        for (size_t paramNumber = params.size(); paramNumber > numFixedParams; paramNumber--)
        {
            // process optional parameter if it exists
            std::string propName, value;
            if (OptionalParameter(params[paramNumber - 1], propName, value))
            {
                if (EqualInsensitive(propName, "copyFlag", "copy"))
                {
                    if (EqualInsensitive(value, "all"))
                    {
                        copyFlags = CopyNodeFlags::copyNodeAll;
                    }
                    else if (EqualInsensitive(value, "value"))
                    {
                        copyFlags = CopyNodeFlags::copyNodeValue;
                    }
                    else
                    {
                        RuntimeError("Invalid optional parameter value %s in CopyNode(), valid values are copyFlag=(all|value)", value.c_str());
                    }
                }
                else
                {
                    RuntimeError("Invalid optional parameter to Copy, %s\n valid optional parameters: copyFlag=(all|value)", propName.c_str());
                }
            }
        }

        return copyFlags;
    }

    // FileParse - parse at the file level, can be overridden for "section of file" behavior
    // stringParse - file concatentated as a single string
    void FileParse(const std::string& stringParse)
    {
        ConfigParameters sections(stringParse);
        bool loadOrEditFound = false;

        // load all the sections that we want (macros)
        if (sections.Exists("load"))
        {
            m_ndlScript.FileParse(stringParse);
            loadOrEditFound = true;
        }

        // load and then execute
        if (sections.Exists("edit"))
        {
            auto config = ConfigArray(sections("edit"));
            for (int i = 0; i < config.size(); ++i)
            {
                Parse(sections(config[i]));
            }
            loadOrEditFound = true;
        }

        // didn't find any of the tags, so just parse the whole thing as a script
        if (!loadOrEditFound)
        {
            // surround text in braces so we parse correctly
            std::string textInBraces = "[ " + stringParse + " ]";
            Parse(textInBraces);
        }
    }

    // CallStringParse - parse the string description of a call sequence
    // token - [in] string description of the call
    // nameFunction - [out] name of the function being called
    // params - [out] parameters to the function, set to empty string if no parameters
    void CallStringParse(const std::string& token, std::string& nameFunction, std::string& params)
    {
        auto paramStart = token.find_first_of(OPENBRACES);
        if (paramStart == npos)
            RuntimeError("Invalid macro/function call can not be parsed: %s\n", token.c_str());
        nameFunction = token.substr(0, paramStart);
        Trim(nameFunction);
        params = token.substr(paramStart);
    }

    bool OptionalParameter(const std::string& token, std::string& name, std::string& value)
    {
        auto foundEqual = token.find_first_of('=');
        bool optional = (foundEqual != npos);
        if (optional)
        {
            name = token.substr(0, foundEqual);
            Trim(name);
            value = token.substr(foundEqual + 1);
            Trim(value);
            TrimQuotes(value);
        }
        return optional;
    }

    // EvaluateNDLSnippet - evaluate the passed snippet of NDL into a computational network
    // script - [in] text of the NDL snippet
    // network - [in/out] computation network to insert NDL into
    void EvaluateNDLSnippet(const ConfigValue& script, ComputationNetworkPtr network)
    {
        NDLUtil<ElemType> ndlUtil(network);
        ndlUtil.ProcessNDLConfig(script);
    }

    // HandleNDLInline - Handle in-line NDL commands
    // stringParse - string we are parsing
    // tokenStart - where the current token starts
    // tokenEnd - end of the current token
    // returns: end of handled text
    std::string::size_type HandleNDLInline(const std::string& stringParse, std::string::size_type tokenStart, std::string::size_type tokenEnd)
    {
        if (m_netNdlDefault->ndl == nullptr)
        {
            m_netNdlDefault->ndl = new NDLScript<ElemType>();
            m_netNdlDefault->ndl->SetMacroDefinitionsAllowed(false);
            m_netNdlDefault->ndl->SetComputationNetwork(m_netNdlDefault->cn);
        }
        std::string::size_type ret = m_netNdlDefault->ndl->ParseValue(stringParse, tokenStart, tokenEnd);
        NDLUtil<ElemType> ndlUtil(m_netNdlDefault->cn);
        ndlUtil.ProcessNDLScript(m_netNdlDefault->ndl, ndlPassInitial, m_netNdlDefault->lastNode);
        return ret;
    }

    // parse a 'key=value' pair and create the appropriate node for what was seen
    // key=Function(x,y,z) - function with return
    // Function(x,y,z) - function with no return
    // HDim = 256 - inline NDL command
    // model1=[...] - Embedded NDL script
    std::string::size_type ParseValue(const std::string& stringParse, std::string::size_type tokenStart, std::string::size_type tokenEnd)
    {
        // skip leading spaces
        tokenStart = stringParse.find_first_not_of(" \t", tokenStart);
        auto keyEnd = stringParse.find_first_of(OPENBRACES "=", tokenStart);
        bool equalFound = (keyEnd != npos && keyEnd < tokenEnd && stringParse[keyEnd] == '=');
        std::string key = stringParse.substr(tokenStart, keyEnd - tokenStart);
        Trim(key);

        // key=Function(x,y,z) - function with return
        // HDim=256 - NDL inline
        // model1=[...] - Embedded NDL script
        if (equalFound)
        {
            size_t tokenStartNew = keyEnd + 1;
            if (!(tokenStartNew < tokenEnd))
                RuntimeError("Equal at the end of line not allowed");
            std::string rightValue = stringParse.substr(tokenStartNew, tokenEnd - tokenStartNew);
            Trim(rightValue);

            auto foundBrace = rightValue.find_first_of(OPENBRACES);
            // HDim=4096 - NDL command
            if (foundBrace == npos)
            {
                if (!m_netNdlDefault)
                    RuntimeError("NDL Command cannot be executed until default model is established, cannot set '%s' without a default mode\n Try calling SetDefaultModel(model) before any NDL statement are embedded\n", key.c_str());
                HandleNDLInline(stringParse, tokenStart, tokenEnd);
            }
            else // createModel, loadModel, or loadNDL
            {
                // model1=[...] - Embedded NDL script
                if (0 == foundBrace)
                {
                    ComputationNetworkPtr cn = make_shared<ComputationNetwork>();
                    EvaluateNDLSnippet(rightValue, cn);
                    OverrideModelNameAndSetDefaultModel(cn, key);
                }
                // key=Function(x,y,z) - function with return, only LoadModel and CreateModel return a value in MEL
                // all other commands will be interpretted as NDL
                else
                {
                    std::string functionName;
                    std::string paramList;
                    CallStringParse(rightValue, functionName, paramList);
                    ConfigParamList params(paramList);

                    if (EqualInsensitive(functionName, "CreateModel"))
                    {
                        params.insert(params.begin(), key);
                        CallFunction("CreateModelWithName", params);
                    }
                    else if (EqualInsensitive(functionName, "LoadModel"))
                    {
                        params.insert(params.begin(), key);
                        CallFunction("LoadModelWithName", params);
                    }
                    else
                    { // not a MEL command, so pass it on to NDL
                        if (!m_netNdlDefault)
                            RuntimeError("NDL Command cannot be executed until default model is established, cannot set '%s' without a default mode\n Try calling SetDefaultModel(model) before any NDL statement are embedded\n", key.c_str());
                        HandleNDLInline(stringParse, tokenStart, tokenEnd);
                    }
                }
            }
            return tokenEnd;
        }
        // Function(x,y,z) - function with no return
        else
        {
            std::string value = stringParse.substr(tokenStart, tokenEnd - tokenStart);
            if (keyEnd > tokenEnd)
                RuntimeError("Invalid line, expecting function call, %s", value.c_str());
            std::string functionName;
            std::string paramList;
            // Function(x,y,z) - function with no return
            CallStringParse(value, functionName, paramList);
            ConfigParamList params(paramList);
            CallFunction(functionName, params);
        }
        return tokenEnd;
    }
};
} } }
