//
// <copyright file="PTaskGraphBuilder.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#ifdef USE_PTASK
#include "PTask.h"
#endif

#include <string>
//#include <cuda_runtime.h>
#include "ComputationNetwork.h"
#include "ComputationNode.h"
#include "PTaskGraphBuilder.h"


namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
TaskDescriptor<ElemType>::TaskDescriptor(
    const ComputationNode<ElemType>* node,
    TaskType taskType,
    size_t input
    ) 
    : m_node(const_cast<ComputationNodePtr>(node)), m_taskType(taskType)
{
    std::string taskName;

    switch(taskType)
    {
    case taskUpdate: // Update task
        {
        m_taskName = "Update_" + msra::strfun::utf8(node->NodeName());
        break;
        }
    case taskEvaluate: // EvaluateThisNode() task
        {
        m_taskName = msra::strfun::utf8(node->NodeName());
        m_taskName = "Forward_" + m_taskName;
        break;
        }
    case taskComputeInputPartial: // ComputeInputPartial() task
        m_taskName = msra::strfun::utf8(node->NodeName());
        char number[2];
        sprintf(number, "%lu", input);
        m_taskName = "Backprop" + std::string(number) + "_" + m_taskName;
        break;
    case taskEvaluateRecurrent: // EvaluateThisNode() Recurrent task
    case taskComputeInputPartialRecurrent: // ComputeInputPartial() Recurrent task
        throw runtime_error("Recurrent not yet implemented");
        break;
    }
    m_task = nullptr;
}

template<class ElemType>
TaskDescriptor<ElemType>::~TaskDescriptor()
{
}

// GradientParam - Get a Gradient Param
// index - index of input, or -1 for current node gradient
// options - param options
template<class ElemType>
ParamData<ElemType>* TaskDescriptor<ElemType>::GradientParam(int index, UINT options, ElemType initValue)
{
    ComputationNodePtr inputNode = index<0?m_node:m_node->Inputs(index);
    std::string valueName = msra::strfun::utf8(inputNode->NodeName()) + "_GradientValues";
    ParamData<ElemType>* pd = new ParamData<ElemType>(paramTypeMatrix, valueName, &inputNode->GradientValues(), options);
    if (!(options&paramOptionsNoPush))
        m_paramData.push_back(pd);
    if (options&paramOptionsInitialize)
        pd->SetInitialize(initValue);
    return pd;
}

// FunctionParam - Insert a Function Param into the parameter list
// index - index of input, or -1 for current node gradient
// options - param options
template<class ElemType>
ParamData<ElemType>* TaskDescriptor<ElemType>::FunctionParam(int index, UINT options)
{
    ComputationNodePtr inputNode = index<0?m_node:m_node->Inputs(index);
    std::string valueName = msra::strfun::utf8(inputNode->NodeName()) + "_FunctionValues";
    ParamData<ElemType>* pd = new ParamData<ElemType>(paramTypeMatrix, valueName, &inputNode->FunctionValues(), options);
    if (!(options&paramOptionsNoPush))
        m_paramData.push_back(pd);
    return pd;
}

// MatrixParam - Insert a Matrix Param into the parameter list
// matrix - matrix we are using as a parameter
// name - name to be used for this parameter, will have node name prepended
// options - param options
template<class ElemType>
ParamData<ElemType>* TaskDescriptor<ElemType>::MatrixParam(const Matrix<ElemType>& matrix, const std::string& name, UINT options)
{
    std::string valueName = msra::strfun::utf8(m_node->NodeName()) + "_" + name;
    ParamData<ElemType>* pd = new ParamData<ElemType>(paramTypeMatrix, valueName, (void *)&matrix, options);
    if (!(options&paramOptionsNoPush))
        m_paramData.push_back(pd);
    return pd;
}

// Param - Insert a Param into the parameter list
// paramType - parameter type for this parameter
// name - name to be used for this parameter, will have node name prepended
// options - param options
template<class ElemType>
ParamData<ElemType>* TaskDescriptor<ElemType>::Param(ParamType paramType, const std::string& name, UINT options, void* data)
{
    std::string valueName = msra::strfun::utf8(m_node->NodeName()) + "_" + name;
    ParamData<ElemType>* pd = new ParamData<ElemType>(paramType, valueName, data, options);
    if (!(options&paramOptionsNoPush))
        m_paramData.push_back(pd);
    return pd;
}

// if PTask is not being used, comment it all out
#ifndef USE_PTASK
template<class ElemType>
void PTaskGraphBuilder<ElemType>::PushActualMBSize(const std::list<ComputationNodePtr>& /*learnableNodes*/, size_t /*actualMBSize*/, CONTROLSIGNAL /*signal=DBCTLC_NONE*/)
{}
template<class ElemType>
void PTaskGraphBuilder<ElemType>::PushData(std::map<std::wstring, Matrix<ElemType>*>& /*data*/, CONTROLSIGNAL /*signal=DBCTLC_NONE*/)
{}
template<class ElemType>
ElemType PTaskGraphBuilder<ElemType>::GetValue(ComputationNode<ElemType>*)
{ return ElemType(1);}
template<class ElemType>
void PTaskGraphBuilder<ElemType>::GetValue(ComputationNode<ElemType>* /*node*/, Matrix<ElemType>& /*matTo*/)
{}
template<class ElemType>
PTaskGraphBuilder<ElemType>::PTaskGraphBuilder() 
{}
template<class ElemType>
PTaskGraphBuilder<ElemType>::~PTaskGraphBuilder() 
{}
template<class ElemType>
void PTaskGraphBuilder<ElemType>::BuildFromComputationNetwork(ComputationNetwork<ElemType>*)
{}
template<class ElemType>
void PTaskGraphBuilder<ElemType>::StartPTaskGraph()
{}
template<class ElemType>
void PTaskGraphBuilder<ElemType>::UpdateParameters(void* /*sgd*/, const ElemType /*learnRatePerSample*/, const size_t /*expectedMBSize*/)
{}
template<class ElemType>
void PTaskGraphBuilder<ElemType>::PushMatrix(const Matrix<ElemType>& /*matrix*/, Channel* /*channel*/, CONTROLSIGNAL signal)
{
    signal;
}

#else //defined(USE_PTASK)
#pragma comment(lib, "ptask.lib")

DatablockTemplate* TaskDescriptor<float>::s_descriptorTemplate = NULL;
DatablockTemplate* TaskDescriptor<double>::s_descriptorTemplate = NULL;

template<class ElemType>
void TaskDescriptor<ElemType>::ConfigureInputsAndOutputs(UINT& uidCounter, std::map<const std::string, Port*>& valueNameToProducerPortMap)
{
    // get the counts of ports
    m_numInputPorts = 0;
    m_numOutputPorts = 0;
    for (ParamData<ElemType>* param : m_paramData)
    {
        if (param->options & paramOptionsInput)
            m_numInputPorts++;
        if (param->options & paramOptionsOutput)
            m_numOutputPorts++;
    }
    m_numInputPorts++; // add an extra one for the TaskDescriptor pointer (always first parameter)
    m_inputPorts = new PTask::Port*[m_numInputPorts];
    m_outputPorts = new PTask::Port*[m_numOutputPorts];
    int paramCount[PTask::PORTTYPE::OUTPUT_PORT+1] = {0};

    // Add the TaskDescriptor parameter that all methods will have
    if (s_descriptorTemplate == NULL)
    {
        BUFFERDIMENSIONS dims(sizeof(TaskDescriptor<ElemType>*));
        s_descriptorTemplate = Runtime::GetDatablockTemplate("TaskDescriptorTemplate", &dims, 1, FALSE, TRUE); // TODO which override? flag values?
    }
    Port* port = CreatePortForTemplate(s_descriptorTemplate, PTask::PORTTYPE::STICKY_PORT, std::string("TaskDescriptor"), 0, -1, false/*gpuBuffer*/, uidCounter, valueNameToProducerPortMap);
    m_inputPorts[paramCount[PTask::PORTTYPE::INPUT_PORT]++] = port;
    void* pv = this;
    Datablock* pblock = PTask::Runtime::AllocateDatablock(s_descriptorTemplate, (void *)&pv, sizeof(this), NULL);
    port->SetPermanentBlock(pblock);
    pblock->Release();

    // create the ports for the other parameters
    int paramNum = 0;
    for (ParamData<ElemType>* param : m_paramData)
    {
        Port* port = NULL;

        // loop for support of in/out ports
        for (UINT portLoop = PTask::PORTTYPE::INPUT_PORT;
            portLoop <= PTask::PORTTYPE::OUTPUT_PORT;
            portLoop = (PTask::PORTTYPE)((UINT)portLoop++))
        {
            UINT portType = portLoop;
            int outputPort = -1;
            if (portType == PTask::PORTTYPE::INPUT_PORT)
            {
                // is it some input type?
                if (!(param->options & (paramOptionsInput|paramOptionsInitialize|paramOptionsConstant|paramOptionsTemporary)))
                    continue;
                // for now treat a temporary as a initializer port that takes zero as it's initalizer value, may change later to a pool or something else
                if (param->options & (paramOptionsInitialize|paramOptionsTemporary))
                    portType = PTask::PORTTYPE::INITIALIZER_PORT;

                // Do this to setup an in/out port
                if (param->options & paramOptionsOutput)
                    outputPort = paramCount[PTask::PORTTYPE::OUTPUT_PORT];
            }
            // check for output port
            else if (!(param->options & paramOptionsOutput))
            {
                continue;
            }

            // get the dimensions to use for the data template
            BUFFERDIMENSIONS dims;

            // determine name for port
            char buf[10];
            _itoa_s(paramNum, buf, 10);
            std::string name = m_taskName + "." + buf;
            bool gpuBuffer = false;

            // get size and suffix for name based on type
            switch(param->type)
            {
            case paramTypeChar:
                dims.Initialize(sizeof(char));
                name += ".char";
                break;
            case paramTypeShort:
                dims.Initialize(sizeof(short));
                name += ".short";
                break;
            case paramTypeLong:
                dims.Initialize(sizeof(long));
                name += ".long";
                break;
            case paramTypeLongLong:
                dims.Initialize(sizeof(long));
                name += ".longlong";
                break;
            case paramTypePointer:
            case paramTypeNode: // same as constant pointer
                dims.Initialize(sizeof(void*));
                name += ".pointer";
                break;
            case paramTypeSingle:
                dims.Initialize(sizeof(float));
                name += ".float";
                break;
            case paramTypeDouble:
                dims.Initialize(sizeof(double));
                name += ".double";
                break;
            case paramTypeMatrix:
                {
                Matrix<ElemType>* matrix = (Matrix<ElemType>*)param->assocData;
                dims.Initialize((UINT)matrix->GetNumRows(), (UINT)matrix->GetNumCols(), 1, sizeof(ElemType), (UINT)(sizeof(ElemType)*matrix->GetNumRows() /* ADim.colstride */ ));
                name += ".matrix";
                gpuBuffer = true;
                break;
                }
            default:
                assert(false);
                break;
            }

            // TODO: Any benefit to caching/reusing templates across different tasks? DBN doesn't bother; 
            // PTask doesn't use unique identity, it seems - at least at present. Do it anyway, in case this changes in the future?
            DatablockTemplate* dt = Runtime::GetDatablockTemplate((char *)name.c_str(), &dims, 1, FALSE, TRUE); // TODO which override? flag values?

            // if this is a matrix type we want to add an application context
            if (param->type == paramTypeMatrix)
            {
                dt->SetApplicationContextCallback((LPFNAPPLICATIONCONTEXTCALLBACK)PTaskGraphBuilder<ElemType>::ApplicationContextCallback);
            }

            // create the port, we add 1 to the paramNum because we always have the taskDescriptor as the first parameter to the task, but it's not in the parameter collection
            port = CreatePortForTemplate(dt, portType, param->name, paramNum+1, outputPort, gpuBuffer, uidCounter, valueNameToProducerPortMap);

            // constant params are handled by setting their port as sticky.
            // (Confusingly, they have PORTTYPE::INPUT and not PORTTYPE::STICKY_PORT - the latter are reserved for 
            // immutable constant values that can be passed by value. Rename them to PORTTYPE::CONSTBYVALUE_PORT?)
            if (param->options & paramOptionsConstant)
                port->SetSticky(TRUE);

            // now populate the input/output array we need to use for creating the task
            if (portLoop == PTask::PORTTYPE::OUTPUT_PORT)
                m_outputPorts[paramCount[portLoop]] = port;
            else
                m_inputPorts[paramCount[portLoop]] = port;
            paramCount[portLoop]++;

            // update the parameter values with the port. If it's an in/out port the 'in' side will be saved
            if (param->port == NULL)
                param->port = port;
            if (portLoop == PTask::PORTTYPE::OUTPUT_PORT)
                param->portOut = port;

            // if we are initializing a buffer, do it now, temporaries will init to zero since that is the default init value
            if (param->options & (paramOptionsInitialize|paramOptionsTemporary))
            {
                dt->SetInitialValue(&param->initValue, sizeof(param->initValue), dt->GetDatablockByteCount()/sizeof(param->initValue));
            }
        }
        paramNum++;
    }
}

template<class ElemType>
void TaskDescriptor<ElemType>::CreateTask(Graph* graph)
{
    CompiledKernel * kernel = PTask::Runtime::GetHostFunctionCompiledKernel(
        const_cast<char*>(m_taskName.c_str()), // cast away the const, really a PTask prototype error
        (FARPROC)HostTaskDriver<ElemType>);
    if(kernel == NULL)
    {
        throw exception("Failed to get PTask kernel");
    }

    m_task = graph->AddTask(
        kernel, 
        m_numInputPorts,
        m_inputPorts,
        m_numOutputPorts,
        m_outputPorts,
        (char*)m_taskName.c_str());

    // This is a host task that will launch CUDA kernels on a dependent CUDA-capable accelerator.
    m_task->BindDependentAcceleratorClass(ACCELERATOR_CLASS_CUDA, 1, TRUE);
}

template<class ElemType>
void TaskDescriptor<ElemType>::CreateChannelsForInputs(
    Graph* graph, 
    std::map<const std::string, Port*>& valueNameToProducerPortMap, 
    std::map<const std::string, std::vector<PTask::GraphInputChannel*>*>& inputNameToChannelsMap,
    int verbosity)
{
    int iparam = 0; 
    // start at 1 because we want to skip the TaskDescriptor parameter, which is always 0
    for (UINT i=1; i<m_numInputPorts; i++)
    {
        // find the associated parameter data
        ParamData<ElemType>* param;
        // for (param = m_paramData[iparam]; 
        //     !(param->options & (paramOptionsInput|paramOptionsConstant|paramOptionsInitialize))
        //     && iparam < m_paramData.size();
        //      param = m_paramData[++iparam])
        //         ; 
        for (param = m_paramData[iparam++];
            !(param->options & (paramOptionsInput|paramOptionsConstant|paramOptionsInitialize))
            && iparam < m_paramData.size();
            param = m_paramData[iparam++])
                ;

        Port* destinationPort = m_inputPorts[i];
        std::string valueName = m_inputNames.at(i);
        assert(destinationPort == param->port);

        if (valueName == "TaskDescriptor")
        {
            // nothing to do here
        }
        // if it's an initializer port and doesn't need to retain it's value, we are done
        else if (destinationPort->GetPortType() == PORTTYPE::INITIALIZER_PORT && !(param->options & paramOptionsMaintainValue))
        {
            fprintf(stderr, " %s Initializer port doesn't need a source; port %d of %s\n", 
                valueName.c_str(), i, this->TaskName().c_str());
        }
        else if (valueNameToProducerPortMap.find(valueName) != valueNameToProducerPortMap.end())
        {
            // If the input name has an entry in the value name to port map, create an internal channel
            // from the associated port.
            Port* sourcePort = valueNameToProducerPortMap[valueName];
            std::string channelName = valueName;
            channelName += " -> " + this->TaskName();
            InternalChannel* channel = graph->AddInternalChannel(sourcePort, destinationPort, (char*)channelName.c_str());
            if (verbosity >= 1) fprintf(stderr, 
                "  Internal channel: %s connected to input port %d of %s\n", 
                channelName.c_str(), i, this->TaskName().c_str());
        }
        else if (inputNameToChannelsMap.find(valueName) != inputNameToChannelsMap.end())
        {
            // If it has an entry in the input name to channels map, create a graph input channel
            // for the associated port.
            // TODO Should we be using a multichannel? Consequences of not?
            std::vector<PTask::GraphInputChannel*>* channels = inputNameToChannelsMap[valueName];

            std::string channelName = "InputChannel " + valueName + " -> " + this->TaskName();
            GraphInputChannel* channel = graph->AddInputChannel(destinationPort, (char*)channelName.c_str());
            if (verbosity >= 1) fprintf(stderr, 
                "  Input channel: %s connected to input port %d of %s\n", 
                channelName.c_str(), i, this->TaskName().c_str());
            channels->push_back(channel);
        }
        // constants need to have an input channel too
        else if (param->options & paramOptionsConstant)
        {
            // consider: do we need to save these in a map, if so should we unify this and the input branch of the if?
            std::string channelName = "ConstantValue: " + valueName + " -> " + this->TaskName();
            graph->AddInputChannel(destinationPort, (char*)channelName.c_str());
            if (verbosity >= 1) fprintf(stderr, 
                "  Input channel: %s connected to input port %d of %s\n", 
                channelName.c_str(), i, this->TaskName().c_str());
        }
        else
        {
            bool finalCriteriaNode = false;
            // check to see if this is the "FinalCriteria" task, if so it needs to have a 1x1 matrix with the value 1.0 pushed into it
            if (m_taskType == taskComputeInputPartial && param->type == paramTypeMatrix)
            {
                DatablockTemplate* dataTemplate = param->port->GetTemplate();
                BUFFERDIMENSIONS dim = dataTemplate->GetBufferDimensions();
                if (dim.uiXElements == dim.uiYElements && dim.uiXElements == 0) // empty matrix
                {
                    dim.uiXElements = dim.uiYElements = dim.uiZElements = 1; // 1x1 matrix
                    dim.cbElementStride = dim.cbPitch = sizeof(ElemType);
                    dataTemplate->SetBufferDimensions(dim);
                    param->port->SetSticky(true);
                    ElemType val = (ElemType)1.0;
                    Datablock* pblock = PTask::Runtime::AllocateDatablock(dataTemplate, (void *)&val, sizeof(ElemType), NULL);
                    param->port->SetPermanentBlock(pblock);
                    pblock->Lock();
                    Matrix<ElemType>* matrix = new Matrix<ElemType>(1, 1, (ElemType*)pblock->GetDataPointer(false), matrixFlagDontOwnBuffer, MANAGEDEXTERN);
                    pblock->SetApplicationContext(matrix);
                    pblock->Unlock();
                    pblock->Release();
                    finalCriteriaNode = true;
                    if (verbosity >= 1) fprintf(stderr, 
                        "  BackPropogation source input %s set to value 1.0 on input port %d of %s\n", 
                        valueName.c_str(), i, this->TaskName().c_str());

                }
            }

            if (!finalCriteriaNode && verbosity >= 1) fprintf(stderr, 
                "  ** No source for input %s to connect to input port %d of %s\n", 
                valueName.c_str(), i, this->TaskName().c_str());
        }
    }
}

//CreateInitializerChannel - Create the initialization channel for a port
// graph - graph to add channel to
// port - port we are adding the initialization channel to
// matrix - initial value to use
// name - name of the init channel
template<class ElemType>
void TaskDescriptor<ElemType>::CreateInitializerChannel(
    Graph* graph,
    Port* port,
    Matrix<ElemType>& matrix,
    const std::string& name
    )
{
    port->GetTemplate()->SetInitialValue(&matrix(0,0), (UINT)(matrix.GetNumElements()*sizeof(ElemType)), 1);
    // if we already have another main channel defined (likely) move it over to the control channel
    if (port->GetChannelCount() > 0)
    {
        Channel* mainChannel = port->GetChannel(0);
        port->UnbindChannel(0);
        port->BindControlChannel(mainChannel);
    }
    Channel* pInitChannel = graph->AddInitializerChannel(port, (char*)name.c_str());
    pInitChannel->SetPredicationType(CE_DST, CGATEFN_OPEN_ON_BOF);
    pInitChannel->SetInitialPropagatedControlSignal(DBCTLC_BOF);
}

// FindEmptyOutPorts - find unconnected output ports and hook up "bit buckets"
// this is necessary because some values are produced in a forward pass that may not be consumed by the back pass, and need to be "capped"
template<class ElemType>
void TaskDescriptor<ElemType>::FindEmptyOutPorts(Graph* graph)
{
    Task* pTask = this->m_task;
    std::map<UINT, Port*>* oPorts = pTask->GetOutputPortMap();
    for(std::map<UINT, Port*>::iterator pi=oPorts->begin(); pi!=oPorts->end(); pi++) 
    {
        OutputPort * pOPort = reinterpret_cast<OutputPort*>(pi->second);
        if(pOPort->GetChannelCount() == 0 && 
           pOPort->GetControlChannelCount() == 0) 
        {
            // unbound output channel! plug it up
            std::string name = pTask->GetTaskName();
            name += "-unused";
            GraphOutputChannel* pOutChannel = graph->AddOutputChannel(pOPort, (char*)name.c_str());
            pOutChannel->SetPredicationType(CE_SRC, CGATEFN_DEVNULL);
        }
    }
}

// CreateBackAndInitChannel - Create back channels and initialization channels 
// graph - graph to add the elements to
// outputNameToChannelsMap - map from name to output channel
template<class ElemType>
void TaskDescriptor<ElemType>::CreateBackAndInitChannel(Graph* graph, std::map<const std::string, PTask::GraphOutputChannel*>& outputNameToChannelsMap)
{
    const vector<ParamData<ElemType>*>& params = this->GetParameters();
    for (const ParamData<ElemType>* param : params)
    {
        // initialize the values at the beginning, copy from source matrix
        if ((param->options & (paramOptionsInitOnBOF | paramOptionsMaintainValue | paramOptionsInitalValuesOnDestinations)) && param->assocData != NULL)
        {
            Matrix<ElemType>* matrix = (Matrix<ElemType>*)param->assocData;
            if (matrix != NULL)
            {
                for (int i=0; i < param->portOut->GetChannelCount();i++)
                {
                    Channel* channel = param->portOut->GetChannel(i);
                    Port* portDest = channel->GetBoundPort(CE_DST);

                    // if we aren't putting initalizers on all destination ports, only MaintainValue on in/out parameters
                    if (!(param->options & paramOptionsInitalValuesOnDestinations)
                        && !portDest->IsInOutParameter())
                        continue;

                    // initialize all ports that get data from this port
                    std::string name(channel->GetName());
                    name.append("-Init");
                    CreateInitializerChannel(graph, portDest, *matrix, name.c_str());
                }
            }
        }

        // check for if we need to save a value on output
        if (param->options & paramOptionsSaveOnEOF)
        {
            std::string nameOutChannel = param->name;
            GraphOutputChannel* outChannel = graph->AddOutputChannel(param->portOut, (char*)nameOutChannel.c_str());
            outChannel->SetPredicationType(CE_SRC, CGATEFN_OPEN_ON_EOF);
            outputNameToChannelsMap[nameOutChannel] = outChannel;
        }
    }
}

// PushActualMBSize - push ActualMBSize into the graph
// learnableNodes - List of learnable nodes that need ActualMBSize
// actualMBSize - actual Minibatch Size
// signal - the PTask DataBlock signal to attach to this data
template<class ElemType>
void PTaskGraphBuilder<ElemType>::PushActualMBSize(const std::list<ComputationNodePtr>& learnableNodes, size_t actualMBSize, PTask::CONTROLSIGNAL signal/*=DBCTLC_NONE*/)
{
    for (ComputationNodePtr node : learnableNodes)
    {
        std::string inputName = msra::strfun::utf8(node->NodeName())+"_actualMBSize";
        auto iter = m_inputNameToChannelsMap.find(inputName);
        if (iter == m_inputNameToChannelsMap.end())
            throw std::runtime_error("input channel not created for actualMBSize");
        std::vector<PTask::GraphInputChannel*>* channels = iter->second;
        for (PTask::GraphInputChannel* channel : *channels)
        {
            Datablock* pblock = PTask::Runtime::AllocateDatablock(channel->GetTemplate(), (void *)&actualMBSize, sizeof(size_t), channel, PT_ACCESS_DEFAULT, signal);
            channel->Push(pblock);
            pblock->Release();
        }
    }
}

// PushData - push data into the graph
// data - the name->matrix map<> returned from datareaders
// signal - the PTask DataBlock signal to attach to this data
template<class ElemType>
void PTaskGraphBuilder<ElemType>::PushData(std::map<std::wstring, Matrix<ElemType>*>& data, PTask::CONTROLSIGNAL signal/*=DBCTLC_NONE*/)
{
    for (std::pair<std::wstring, Matrix<ElemType>*> pair : data)
    {
        std::string inputName = msra::strfun::utf8(pair.first)+"_FunctionValues";
        auto iter = m_inputNameToChannelsMap.find(inputName);
        if (iter == m_inputNameToChannelsMap.end())
            throw std::runtime_error("input channel not created for data matrix");
        std::vector<PTask::GraphInputChannel*>* channels = iter->second;
        for (PTask::GraphInputChannel* channel : *channels)
        {
            Matrix<ElemType>* matrix = pair.second;
            PushMatrix(*matrix, channel, signal);
        }
    }
}

// PushMatrix - push matrix into the graph
// matrix - the matrix to push into the graph
// channel - channel we are pushing into
// signal - the PTask DataBlock signal to attach to this data
template<class ElemType>
void PTaskGraphBuilder<ElemType>::PushMatrix(const Matrix<ElemType>& matrix, Channel* channel, PTask::CONTROLSIGNAL signal/*=DBCTLC_NONE*/)
{
    void* matrixData = (void*)matrix.BufferPointer();
    size_t sizeData = matrix.BufferSize();
    Datablock* pblock = PTask::Runtime::AllocateDatablock(channel->GetTemplate(), (void *)matrixData, (UINT)sizeData, channel, PT_ACCESS_DEFAULT, signal);
    pblock->Lock();
    Matrix<ElemType>* header = new Matrix<ElemType>(matrix.GetNumRows(), matrix.GetNumCols(), (ElemType*)pblock->GetDataPointer(false), matrixFlagDontOwnBuffer, MANAGEDEXTERN, matrix.NzCount());
    pblock->SetApplicationContext(header);
    pblock->Unlock();
    channel->Push(pblock);
    pblock->Release();
}

// GetValue - Get a single value out of the graph
// node - node (used to get the name) we want to retieve the value from
// returns - single value from the channel named the same as the node
template<class ElemType>
ElemType PTaskGraphBuilder<ElemType>::GetValue(ComputationNode<ElemType>* node)
{
    std::string nameOutChannel = msra::strfun::utf8(node->NodeName());
    GraphOutputChannel* outChannel = m_outputNameToChannelsMap[nameOutChannel];
    Datablock* dataBlock = outChannel->Pull();
    dataBlock->Lock();
    void* data = dataBlock->GetDataPointer(false);
    assert(dataBlock->GetDataBufferLogicalSizeBytes() >= sizeof(ElemType));

    // if we have the matrix header make sure it's a 1x1 matrix
    if (dataBlock->GetApplicationContext() != NULL)
    {
        Matrix<ElemType>* mat = (Matrix<ElemType>*)dataBlock->GetApplicationContext();
        assert(mat->GetNumCols() == mat->GetNumRows());
        assert(mat->GetNumRows() == 1);
    }
    // return the value
    ElemType elem = *(ElemType*)data;
    dataBlock->Unlock();
    return elem;
}

// GetValue - Get a single value out of the graph
// node - node (used to get the name) we want to retieve the value from
// returns - single value from the channel named the same as the node
template<class ElemType>
void PTaskGraphBuilder<ElemType>::GetValue(ComputationNode<ElemType>* node, Matrix<ElemType>& matTo)
{
    std::string nameOutChannel = msra::strfun::utf8(node->NodeName()) + "_FunctionValues";
    if (m_outputNameToChannelsMap.find(nameOutChannel) == m_outputNameToChannelsMap.end())
        throw std::logic_error("GetValue: name not found in output channel map");
    GraphOutputChannel* outChannel = m_outputNameToChannelsMap[nameOutChannel];
    Datablock* dataBlock = outChannel->Pull();
    dataBlock->Lock();
    void* data = dataBlock->GetDataPointer(false);

    // if we have the matrix header make sure it is sane and copy it over
    if (dataBlock->GetApplicationContext() != NULL)
    {
        Matrix<ElemType>* mat = (Matrix<ElemType>*)dataBlock->GetApplicationContext();
        assert (dataBlock->GetDataBufferLogicalSizeBytes() >= mat->BufferSize());
        matTo = *mat;
    }
    dataBlock->Unlock();
}


// CreateOutputChannels - Create the output channels for the graph
template<class ElemType>
void PTaskGraphBuilder<ElemType>::CreateOutputChannels(const vector<ComputationNodePtr>& nodes)
{
    for (ComputationNode<ElemType>* node : nodes)
    {
        std::string nameOutChannel = msra::strfun::utf8(node->NodeName());
        std::string name = nameOutChannel+"_FunctionValues";
        Port* port = m_valueNameToProducerPortMap[name];
        GraphOutputChannel* outChannel = m_PTaskGraph->AddOutputChannel(port, (char*)nameOutChannel.c_str());
        m_outputNameToChannelsMap[nameOutChannel] = outChannel;
    }
}

// CreatePropogationPath - Create the path for the control signals to follow in the graph
template<class ElemType>
void PTaskGraphBuilder<ElemType>::CreatePropogationPath()
{
    Graph* graph = m_PTaskGraph;
    if (m_verbosity >= 1) fprintf(stderr, "\nCreating Propogation Path through graph...\n");

    // use the VERY COOL auto propogation path creator
    m_PTaskGraph->SynthesizeControlPropagationPaths(DBCTLC_EOF);
}

template<class ElemType>
Port* TaskDescriptor<ElemType>::CreatePortForTemplate(DatablockTemplate* dt,
    UINT portType, 
    std::string& valueName, 
    UINT portIndex, UINT inoutPort, bool gpuBuffer, 
    UINT& uidCounter,
    std::map<const std::string, Port*>& valueNameToProducerPortMap
    )
{
    // TODO Either get variable bindings for each ComputationNode sub-type, or standardize on a fixed set?
    // std::string variableBinding = ???
    // TODO For now, using the value name as the variable binding, as useful for debugging.
    Port* port = PTask::Runtime::CreatePort(
        (PTask::PORTTYPE)portType, dt, uidCounter++, (char*)valueName.c_str(), portIndex, inoutPort); // TODO InOutParamPortIndex should be -1? Other params?

    // Set this port to be dependent on CUDA, which means all the buffers will stay on the CUDA side.
    // Note that the index here is _not_ the accelerator number and must always be 0. Affinity is configured elsewhere.
    // TODO: needs to change for host-side processes if we support them
    if (gpuBuffer)
    {
        port->BindDependentAccelerator(ACCELERATOR_CLASS_CUDA, 0);
    }

    // Store the port, and also the name of the value it corresponds to.
    switch(portType)
    {
    case PTask::PORTTYPE::OUTPUT_PORT:
        m_outputNames.push_back(valueName);

        // In the case of output ports, this port is the producer of this value.
        // Store the name-port association globally, so that consumers can connect to it.
        if (valueNameToProducerPortMap.find(valueName) != valueNameToProducerPortMap.end())
        {
            throw exception("Each value name may only be produced by one task/port");
        }
        valueNameToProducerPortMap[valueName] = port;
        break;
    case PTask::PORTTYPE::STICKY_PORT:
    case PTask::PORTTYPE::INPUT_PORT:
    case PTask::PORTTYPE::INITIALIZER_PORT:
        m_inputNames.push_back(valueName);
        break;
    default:
    case PTask::PORTTYPE::META_PORT:
        throw exception("Unsupported port type");
        break;
    }
    return port;
}

template<class ElemType>
PTaskGraphBuilder<ElemType>::PTaskGraphBuilder() 
{
    // Level of console logging, useful for development/debugging.
    // 0 = silent; 1 = summary; 2 = debug. 
    // m_verbosity = 1;
    m_verbosity = 2;

    m_portUIDCounter = 0;
    PTask::Runtime::SetUseOpenCL(FALSE);
    PTask::Runtime::SetUseDirectX(FALSE);
    PTask::Runtime::SetUseCUDA(TRUE);

    // set the watchdog, so we can tell where a graph is hanging, if it does
    PTask::Runtime::SetUseGraphMonitorWatchdog(TRUE);
    PTask::Runtime::SetDispatchWatchdogThreshold(45000); // inital read of data can take a little time

    //PTask::Runtime::SetDispatchLoggingEnabled(TRUE);

    // CJR: 6/19: PTask tries to keep a default cuda context current on 
    // threads that are likely to make CUDA API calls to limit common case 
    // overheads for device context push/pop/attach/detach. This gets messed up
    // when the application makes it's own context either explicitly or implicitly 
    // (like CNTK) through calls to cuda runtime APIS. Inform PTask that CNTK does this:
    PTask::Runtime::SetApplicationThreadsManagePrimaryContext(TRUE);
       
    // now limit PTask to only use the Accelerators we want it to
    LimitAccelerators();

    // time to initalize the runtime
    PTask::Runtime::Initialize();

    m_PTaskGraph = PTask::Runtime::CreateGraph(); // TODO Add graph name?
    m_cn = NULL;
}

// destructor: stops and teardown the graph
template<class ElemType>
PTaskGraphBuilder<ElemType>::~PTaskGraphBuilder() 
{
    if (m_PTaskGraph == nullptr)
        return;
    if (m_PTaskGraph->IsRunning())
        m_PTaskGraph->Stop();
    m_PTaskGraph->Teardown();
}

template<class ElemType>
void PTaskGraphBuilder<ElemType>::CreateTaskDescriptorForNode(ComputationNode<ElemType>* node, TaskType taskFlag)
{
    switch(taskFlag)
    {
    case taskEvaluate: // EvaluateThisNode() task
        {
        TaskDescriptorPtr taskDescriptor = node->GetPTaskDescriptor(taskFlag);
        m_taskNameToTaskDescriptorMap[taskDescriptor->TaskName()] = taskDescriptor;
        break;
        }
    case taskComputeInputPartial: // ComputeInputPartial() task
        for (int i=0; i < node->ChildrenSize(); i++)
        {
            // if it's an input, we don't need to do a gradient
            if (!node->Inputs(i)->NeedGradient())
                continue;
            TaskDescriptorPtr taskDescriptor = node->GetPTaskDescriptor(taskFlag, i);
            std::string name = taskDescriptor->TaskName();
            if (m_taskNameToTaskDescriptorMap.find(name) != m_taskNameToTaskDescriptorMap.end())
                throw logic_error("Add support for multiple Partial derivatives to PTask");
            m_taskNameToTaskDescriptorMap[name] = taskDescriptor;
        }
        break;
    case taskUpdate:
        {
        TaskDescriptorPtr taskDescriptor = node->GetPTaskDescriptor(taskFlag);
        m_taskNameToTaskDescriptorMap[taskDescriptor->TaskName()] = taskDescriptor;
        break;
        }
    case taskEvaluateRecurrent: // EvaluateThisNode() Recurrent task
    case taskComputeInputPartialRecurrent: // ComputeInputPartial() Recurrent task
        throw runtime_error("Recurrent not yet implemented");
        break;
    }
}


// CreateInputName - create a entry in the input name to channels map
// inputName - name for the entry
template<class ElemType>
void PTaskGraphBuilder<ElemType>::CreateInputName(std::string inputName)
{
    // But do create an (empty) entry for it in input name to channels map.
    // Will populate later with graph input channels.
    if (m_inputNameToChannelsMap.find(inputName) != m_inputNameToChannelsMap.end())
    {
        throw exception("Input names must be unique");
    }
    std::vector<PTask::GraphInputChannel*>* channels = new std::vector<PTask::GraphInputChannel*>();
    m_inputNameToChannelsMap[inputName] = channels;
}

// Create descriptors to model PTask tasks that ComputationNodes will be mapped to.
template<class ElemType>
void PTaskGraphBuilder<ElemType>::CreateTaskDescriptorsForComputationNodes()
{
    if (m_verbosity >= 1) fprintf(stderr, "\nCreating PTask tasks for ComputationNodes ...\n");
    for (auto nodeIter=m_computationNodes.begin(); nodeIter != m_computationNodes.end(); nodeIter++)
    {
        ComputationNodePtr node = *nodeIter;
        std::wstring opName = node->OperationName();

        if (m_verbosity >= 1) fprintf(stderr, "  %ls(%ls): ",
            opName.c_str(), node->NodeName().c_str());

        // Learnable parameter node types.
        if (opName == LearnableParameter<ElemType>::TypeName())
        {
            // if we are computing the gradient
            if (node->NeedGradient())
            {
                if (m_verbosity >= 1) fprintf(stderr, "Creating LearnableParameter Tasks\n");
                CreateTaskDescriptorForNode(node, taskUpdate);
                std::string inputName = msra::strfun::utf8(node->NodeName())+"_actualMBSize";
                CreateInputName(inputName);
            }
        }
        // Input node types
        else if (opName == InputValue<ElemType>::TypeName() || node->RequirePreCompute())
        {
            // Don't create any tasks for an input node ...
            if (m_verbosity >= 1) fprintf(stderr, "Input/Precompute node requires no tasks\n");
            std::string inputName = msra::strfun::utf8(node->NodeName())+"_FunctionValues";
            CreateInputName(inputName);
        }
        // Forward only node types.
        else if (
            opName == PerDimMeanVarNormalizationNode<ElemType>::TypeName() ||
            opName == PerDimMeanVarDeNormalizationNode<ElemType>::TypeName() ||
            opName == ErrorPredictionNode<ElemType>::TypeName())
        {
            if (m_verbosity >= 1) fprintf(stderr, "Creating forward task only\n");
            CreateTaskDescriptorForNode(node, taskEvaluate);
        }
        // Regular computation node types.
        else if (
            opName == CrossEntropyWithSoftmaxNode<ElemType>::TypeName() ||
            opName == MatrixL1RegNode<ElemType>::TypeName() ||
            opName == MatrixL2RegNode<ElemType>::TypeName() ||
            opName == SquareErrorNode<ElemType>::TypeName() ||
            opName == ConvolutionNode<ElemType>::TypeName() ||
            opName == MaxPoolingNode<ElemType>::TypeName() ||
            opName == CosDistanceNode<ElemType>::TypeName() ||
            opName == ElementTimesNode<ElemType>::TypeName() ||
            opName == DiagTimesNode<ElemType>::TypeName() ||
            opName == DropoutNode<ElemType>::TypeName() ||
            opName == ExpNode<ElemType>::TypeName() ||
            opName == LogNode<ElemType>::TypeName() ||
            opName == CosineNode<ElemType>::TypeName() ||
            opName == MinusNode<ElemType>::TypeName() ||
            opName == NegateNode<ElemType>::TypeName() ||
            opName == PlusNode<ElemType>::TypeName() ||
            opName == RectifiedLinearNode<ElemType>::TypeName() ||
            opName == ScaleNode<ElemType>::TypeName() ||
            opName == SigmoidNode<ElemType>::TypeName() ||
            opName == SoftmaxNode<ElemType>::TypeName() ||
            opName == SumElementsNode<ElemType>::TypeName() ||
            opName == TanhNode<ElemType>::TypeName() ||
            opName == TimesNode<ElemType>::TypeName())
        {
            if (m_verbosity >= 1) fprintf(stderr, "Creating forward and back tasks\n");
            CreateTaskDescriptorForNode(node, taskEvaluate);
            CreateTaskDescriptorForNode(node, taskComputeInputPartial);
        }
        else
        {
            fprintf(stderr, "PTaskGraphBuilder does not (yet) support ComputationNode type %ls.\n",
                opName.c_str());
            throw exception("Unsupported computation node type");
        }
    }
}

// UpdateParameters - Update the parameters that may change by epoch
// sgd - SGD pointer (to access constants in the class)
// learnRatePerSample - learning rate per sample for this epoch
// expectedMBSize - minibatch size for this epoch
template<class ElemType>
void PTaskGraphBuilder<ElemType>::UpdateParameters(void* sgd, const ElemType learnRatePerSample, const size_t expectedMBSize)
{
    std::list<ComputationNodePtr> precompNodes = m_cn->GetNodesRequirePreComputation(nullptr, false);
    for (ComputationNodePtr node : precompNodes)
    {
        std::string name = msra::strfun::utf8(node->NodeName()) + "_FunctionValues";
        std::map<const std::string, std::vector<PTask::GraphInputChannel*>*>::iterator iter 
            = m_inputNameToChannelsMap.find(name);
        if (!(iter == m_inputNameToChannelsMap.end()))
        {
            // make sure the matrix is in CPU memory before we add to the graph
            Matrix<ElemType>& mat = node->FunctionValues();
            mat.SetPreferredDeviceId(CPUDEVICE);
            mat.TransferFromDeviceToDevice(mat.GetDeviceId(), CPUDEVICE, true);

            vector<GraphInputChannel*>* inChannels = iter->second;
            for (GraphInputChannel* input : *inChannels)
            {
                PushMatrix(node->FunctionValues(), input);
            }
        }
    }

    for (std::pair<const std::string, TaskDescriptorPtr> pair : m_taskNameToTaskDescriptorMap)
    {
        const TaskDescriptorPtr taskDescriptor = pair.second;
        const std::vector<ParamData<ElemType>*>& params = taskDescriptor->GetParameters();
        ComputationNodePtr node = taskDescriptor->GetNode();

        // update needs the learning Rate and MBSizes, etc. Set them here
        if (taskDescriptor->GetTaskType() == taskUpdate)
        {
            //static void UpdateWeightsS(const SGD* sgd, Matrix<ElemType>& functionValues, Matrix<ElemType>& gradientValues, Matrix<ElemType>& smoothedGradient, 
            //    const ElemType learnRatePerSample, const size_t actualMBSize, const size_t expectedMBSize)
            params[0]->SetConstant((void*)&sgd, sizeof(sgd));
            params[4]->SetConstant((void*)&learnRatePerSample, sizeof(learnRatePerSample));
            assert(sizeof(void*) == sizeof(expectedMBSize));
            params[6]->SetConstant((void*)&expectedMBSize, sizeof(expectedMBSize));
        }
    }
}

// the Host Task driver
template <class ElemType>
static void __stdcall
HostTaskDriver(LPDEPENDENTCONTEXT depContext)
{
    assert(depContext->nArguments > 1); // should always have more than one parameter (the context parameter)
    assert(depContext->ppArguments != NULL);
                
    // look for a dependent binding
    bool hasDependentBinding = false;
    for (int i=1; i < depContext->nArguments; i++)
    {
        if (depContext->pbIsDependentBinding[i])
        {
            hasDependentBinding = true;
            break;
        }
    }
    
    // if we have a dependent binding, need to launch GPU methods
    if (hasDependentBinding) {
        void * pvStream = depContext->pStreams[0];
        cudaStream_t hStream = reinterpret_cast<cudaStream_t>(pvStream);
        size_t device = reinterpret_cast<size_t>(depContext->pDependentDevices[0]);
        onstream override(hStream);
        Datablock** datablocks = depContext->ppDatablocks;

        TaskDescriptor<ElemType>* taskDescriptor = *((TaskDescriptor<ElemType>**) depContext->ppArguments[0]);
        const std::vector<ParamData<ElemType>*>& parameters = taskDescriptor->GetParameters();
        void* newParam[10]; // array to hold parameters
        assert(parameters.size() <= 10);
        assert(depContext->nArguments == parameters.size()+1);
        int argument=1; // first argument will always be the TaskDescriptor, then start regular arguments
        for (int i = 0; i < parameters.size(); i++, argument++)
        {
            const ParamData<ElemType>*  paramData = parameters[i];
            //JC Datablock* datablock = datablocks[i];
            Datablock* datablock = datablocks[argument];
            switch(paramData->type)
            {
            case paramTypeNode: // this parameter is a node pointer, just get it from the taskDescriptor
                // NOTE: this can ONLY be used for accessing constant values in the node, not allowed to change anything.
                newParam[i] = (void*)taskDescriptor->GetNode();
                break;
            case paramTypeMatrix: // the actual datapointer will be a GPU side buffer for the matrix
                {
                Matrix<ElemType>* matrix = (Matrix<ElemType>*)datablock->GetApplicationContext();
                // determine the device for this argument
                if (depContext->pbIsDependentBinding[argument])
                    device = reinterpret_cast<size_t>(depContext->pvDeviceBindings[argument]);
                else
                    device = CPUDEVICE;
                // if there is no matrix associated with the datablock yet, associate one
                if (matrix == NULL || (paramData->options & paramOptionsOutput))
                {
                    Matrix<ElemType> *hostMatrix = (Matrix<ElemType> *)paramData->assocData; // reference the node based matrix
                    bool sparse = hostMatrix->GetMatrixType() == SPARSE;
                    matrix = new Matrix<ElemType>(hostMatrix->GetNumRows(), hostMatrix->GetNumCols(), (ElemType *)depContext->ppArguments[argument], 
                        matrixFlagDontOwnBuffer | (sparse?matrixFormatSparse:matrixFormatDense), device);
                    datablock->SetApplicationContext(matrix);
                }
                else
                {
                    // update the datapointer in case it changed
                    assert(matrix->GetCurrentMatrixLocation() == GPU || matrix->GetCurrentMatrixLocation() == BOTH);
                    matrix->SetValue(matrix->GetNumRows(), matrix->GetNumCols(), (ElemType*)depContext->ppArguments[argument], matrixFlagDontOwnBuffer, device);
                }
                newParam[i] = matrix;
                break;
                }
                // everything else fits into a void* space
            default:
                newParam[i] = *((void**)depContext->ppArguments[argument]);
                break;
            }
        }

#if DUMPOUTPUT
        fprintf(stderr, "%s\n", taskDescriptor->TaskName().c_str());
#endif
        // call the actual function now, must take all pointers/references or integral types that fit into sizeof(void*) bytes to work
        FARPROC function = taskDescriptor->GetFunction();
        typedef void (*func1)(void*);
        typedef void (*func2)(void*, void*);
        typedef void (*func3)(void*, void*, void*);
        typedef void (*func4)(void*, void*, void*, void*);
        typedef void (*func5)(void*, void*, void*, void*, void*);
        typedef void (*func6)(void*, void*, void*, void*, void*, void*);
        typedef void (*func7)(void*, void*, void*, void*, void*, void*, void*);
        typedef void (*func8)(void*, void*, void*, void*, void*, void*, void*, void*);
        typedef void (*func9)(void*, void*, void*, void*, void*, void*, void*, void*, void*);
        switch (parameters.size())
        {
        case 1:
            (*(func1)function)(newParam[0]);
            break;
        case 2:
            (*(func2)function)(newParam[0], newParam[1]);
            break;
        case 3:
            (*(func3)function)(newParam[0], newParam[1], newParam[2]);
            break;
        case 4:
            (*(func4)function)(newParam[0], newParam[1], newParam[2], newParam[3]);
            break;
        case 5:
            (*(func5)function)(newParam[0], newParam[1], newParam[2], newParam[3], newParam[4]);
            break;
        case 6:
            (*(func6)function)(newParam[0], newParam[1], newParam[2], newParam[3], newParam[4], newParam[5]);
            break;
        case 7:
            (*(func7)function)(newParam[0], newParam[1], newParam[2], newParam[3], newParam[4], newParam[5], newParam[6]);
            break;
        case 8:
            (*(func8)function)(newParam[0], newParam[1], newParam[2], newParam[3], newParam[4], newParam[5], newParam[6], newParam[7]);
            break;
        case 9:
            (*(func9)function)(newParam[0], newParam[1], newParam[2], newParam[3], newParam[4], newParam[5], newParam[6], newParam[7], newParam[8]);
            break;

        }

        //TODO: we need to free up all the memory
    } else {
        // in this case the depContext->ppArguments[*]  are host pointers                 
        // call host version if it exists
        assert(false);  // no native version
    }
}

template<class ElemType>
void PTaskGraphBuilder<ElemType>::BuildFromComputationNetwork(ComputationNetwork<ElemType>* cn)
{
    m_cn = cn;
    m_computationNodes = cn->GetAllNodes();

    // set all the input nodes to have the biggest minibatch size we will see in this run
    ComputationNodePtr criteriaNode = cn->FinalCriterionNodes()[0];
    int actualMBSize = cn->GetMaxMBSize();
    for (ComputationNodePtr node : cn->InputNodes(criteriaNode))
    {
        Matrix<ElemType>& matrix = node->FunctionValues();
        matrix.Resize(matrix.GetNumRows(), actualMBSize);
    }
    // validate the network to make sure all the matrices get resized to match the inputs
    cn->ValidateNetwork(criteriaNode);

    // clear the Gradients for all nodes, because this will resize all the gradients to be the proper size
    cn->ClearGradientForAllNodes(criteriaNode);

    this->CreateTaskDescriptorsForComputationNodes();
    this->ConfigureTaskInputsAndOutputs();
    this->CreateTasksFromDescriptors();
    this->CreateChannels();
    this->CreateBackAndInitChannels();
    CreateOutputChannels(cn->FinalCriterionNodes());
    CreateOutputChannels(cn->EvaluationNodes());
    //CreateOutputChannels(cn->OutputNodes());
    this->FindEmptyOutPorts();
    this->CreatePropogationPath();
}

template<class ElemType>
void PTaskGraphBuilder<ElemType>::StartPTaskGraph()
{
    if (!IsRunning())
        StartGraph();
}

// ApplicationContextCallback - Callback to manage the application context
// eCallbackPoint - [in] The point in the datablock's lifecycle at which the callback was called. 
// pDatablock -[in] The datablock being created, cloned or destroyed. 
// ppApplicationContext - [in/out] The application context to be managed. 
// Note: you must register the callback with the datablock template using the following call:
//   pDataTemplate->SetApplicationContextCallback(PTaskGraphBuilder<ElemType>::ApplicationContextCallback);
template<class ElemType>
void __stdcall PTaskGraphBuilder<ElemType>::ApplicationContextCallback(
    APPLICATIONCONTEXTCALLBACKPOINT eCallbackPoint,
    const Datablock * pDatablock,
    void ** ppApplicationContext
    )
{
    switch (eCallbackPoint)
    {
    case CALLBACKPOINT_CREATE:
        // currently we don't have enough information here to create anything
        // so ignore
        break;
    case CALLBACKPOINT_CLONE:
        // the block was cloned, so we need to clone the header also, 
        // for now we will create the header the next time a NULL appcontext comes in
        break;
    case CALLBACKPOINT_DESTROY:
        // this block is being destroyed, delete the matrix header
        Matrix<ElemType>* pmat = (Matrix<ElemType>*)*ppApplicationContext;
        if (pmat != nullptr) // if we have a matrix pointer, free it
            delete pmat;
        *ppApplicationContext = nullptr;
        break;
    }
}

#endif // USE_PTASK

// instantiate classes
template class PTaskGraphBuilder<float>;
template class PTaskGraphBuilder<double>;

template class TaskDescriptor<float>;
template class TaskDescriptor<double>;

//template ParamData<float>* TaskDescriptor<float>::GradientParam(int, UINT,float);
//template ParamData<double>* TaskDescriptor<double>::GradientParam(int,UINT,double);

}}}
