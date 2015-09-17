// main.cpp -- main function for testing config parsing

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "../../BrainScript/BrainScriptEvaluator.h"

using namespace Microsoft::MSR::BS;

#ifndef let
#define let const auto
#endif

#if 0
// notes on integrating
if (config.Exists("NDLNetworkBuilder"))
{
    ConfigParameters configNDL(config("NDLNetworkBuilder"));
    netBuilder = (IComputationNetBuilder<ElemType>*)new NDLBuilder<ElemType>(configNDL);
}
else if (config.Exists("ExperimentalNetworkBuilder"))
{
    ConfigParameters sourceCode(config("ExperimentalNetworkBuilder"));
    // get sourceCode as a nested string that contains the inside of a dictionary (or a dictionary)
    netBuilder = (IComputationNetBuilder<ElemType>*)new ExperimentalNetworkBuilder<ElemType>(sourceCode);
}
// netBuilder is a wrapper with these methods to create a ComputationNetwork:; see NDLNetworkBuilder.h
ComputationNetwork* net = startEpoch < 0 ? netBuilder->BuildNetworkFromDescription() :
    netBuilder->LoadNetworkFromFile(modelFileName);
// LoadNetworkFromFile() -> NDLNetworkBuilder.h LoadFromConfig() 
// -> NDLUtil.h NDLUtil::ProcessNDLScript()
// does multiple passes calling ProcessPassNDLScript()
// -> NetworkDescriptionLanguage.h NDLScript::Evaluate
// which sometimes calls into NDLNodeEvaluator::Evaluate()
// NDLNodeEvaluator: implemented by execution engines to convert script to approriate internal formats
// here: SynchronousNodeEvaluator in SynchronousExecutionEngine.h
// SynchronousNodeEvaluator::Evaluate()   --finally where the meat is
//  - gets parameters from config and translates them into ComputationNode
//    i.e. corrresponds to our MakeRuntimeObject<ComputationNode>()
//  - creates all sorts of ComputationNode types, based on NDLNode::GetName()
//     - parses parameters depending on node type   --this is the NDL-ComputationNode bridge
//     - creates ComputationNodes with an additional layer of wrappers e.g. CreateInputNode()
//     - then does all sorts of initialization depending on mode type
//  - can initialize LearnableParameters, incl. loading from file. WHY IS THIS HERE?? and not in the node??
//  - for standard nodes just creates them by name (like our classId) through m_net.CreateComputationNode()
// tags:
//  - tags are not known to ComputationNode, but to Network
//  - processed by SynchronousNodeEvaluator::ProcessOptionalParameters() to sort nodes into special node-group lists such as m_featureNodes (through SetOutputNode())

// notes:
//  - InputValue nodes are created from 4 different names: InputValue, SparseInputvalue, ImageInput, and SparseImageInput
//  - for SparseInputvalue, it checks against InputValue::SparseTypeName(), while using a hard-coded string for ImageInput and SparseImageInput
//  - there is also SparseLearnableParameter, but that's a different ComputationNode class type
#endif

namespace Microsoft { namespace MSR { namespace ScriptableObjects {
    const ConfigurableRuntimeType * FindExternalRuntimeTypeInfo(const wstring &) { return nullptr; }
}}}

int wmain(int /*argc*/, wchar_t* /*argv*/[])
{
    SomeTests();
}
