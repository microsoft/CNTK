//
// <copyright file="PTaskGraphBuilder.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#ifdef USE_PTASK
#include "PTask.h"
#else
typedef void Port;
typedef void Graph;
typedef void Task;
typedef void Channel;
typedef int CONTROLSIGNAL;
#define DBCTLC_BOF 1
#define DBCTLC_EOF 2
#define DBCTLC_NONE 3
#endif

#ifndef _WIN32          // BUGBUG: fix this once we need it
typedef unsigned int UINT;
typedef long long (*FARPROC)();
#endif

#include <string>
//#include <cuda_runtime.h>

namespace Microsoft { namespace MSR { namespace CNTK {
#if 0   // TODO: where is this used? It creates the dependency on cuda_runtime.h, which we prefer to not have
    extern __declspec(thread) cudaStream_t t_stream;

    // class for stream overrides for PTask
    // auto-class to set stream override inside a function
    // usage at each function that calls CUDA:
    //  onstream override(stream);
    class onstream
    {
        cudaStream_t prevStream;
    public:
        onstream (cudaStream_t stream) { prevStream = GetStream(); SetStream(stream);}
        ~onstream() { SetStream(prevStream); }
    };
#endif

    // any  pointer/reference and all scalar types that fit into 8 bytes can be parameters
    enum ParamType
    {
        paramTypeNone,
        paramTypeMatrix,
        paramTypePointer,
        paramTypeReference = paramTypePointer, // both the same in memory
        paramTypeShort,
        paramTypeInteger,
        paramTypeLong = paramTypeInteger,
        paramTypeSizet,
        paramTypeLongLong = paramTypeSizet,
        paramTypeSingle,
        paramTypeDouble,
        paramTypeChar,
        paramTypeBool,
        paramTypeNode, // pass the node as first parameter, to be used ONLY for CONSTANT values
    };

    enum ParamOptions
    {
        paramOptionsNull = 0, // invalid value
        paramOptionsInput = 1, // and input value for a task
        paramOptionsOutput = 2, // an output value from a task
        paramOptionsTemporary = 4, // for variables only used within one task (and then thrown away)
        paramOptionsInitialize = 8, // initialize buffer with value
        paramOptionsRecurrantIterator = 16, // iterator for recurrancy
        paramOptionsConstant = 32, // constant value
        paramOptionsMaintainValue = 64, // maintain value for in/out parameters, if not specified in/out means pass-through variable
        paramOptionsNoPush = 128, // flag for parameter routines to not push the values
        paramOptionsInitOnBOF = 256, // Initialize the buffer on BOF signal
        paramOptionsSaveOnEOF = 512, // Save on EOF
        paramOptionsInitalValuesOnDestinations = 1024, // Initial values need to be set on all destination ports, needed for Update
    };

    enum TaskType
    {
        taskNull,   // invalid value
        taskEvaluate, // EvaluateThisNode() task
        taskComputeInputPartial, // ComputeInputPartial() task
        taskEvaluateRecurrent, // EvaluateThisNode() Recurrent task
        taskComputeInputPartialRecurrent, // ComputeInputPartial() Recurrent task
        taskUpdate, // update weight matricies
        taskOutput, // output node, copy back to ComputationNode structure
    };


    // ParamData - parameter data, explains the parameter type
    template<class ElemType>
    class ParamData
    {
    public:
        ParamType type; // data type of the parameter
        void *assocData; // associated data (i.e. for Matrix, the matrix is the node this corresponds to)
        Port* port;  // PTask port that is created for this parameter
        Port* portOut; // output port for in/out parameters
        UINT options; // parameter options (see ParamOptions above)
        ElemType initValue; // initialization data for "paramOptionsInitialize"
        std::string name; // name of the parameter

        // constructors
        ParamData(ParamType type, const std::string& name, UINT options) : type(type), name(name), assocData(NULL), port(NULL), portOut(NULL), options(options), initValue(0)
        {}
        ParamData(ParamType type, const std::string& name, void* data, UINT options) : type(type), name(name), assocData(data), port(NULL), portOut(NULL), options(options), initValue(0)
        {}

        // Initialize a parameter with an inital value
        // initValue - value to initialize the port with
        void SetInitialize(ElemType initVal)
        { 
            initValue = initVal;
            options |= paramOptionsInitialize;
        }

        // SetConstant - Set a constant value to a port
        // data - pointer to the data
        // sizeData - size in bytes of the data
        void SetConstant(void* data, int sizeData)
        {
#ifdef USE_PTASK
            GraphInputChannel* channel = (GraphInputChannel*)port->GetChannel(0);
            Datablock* pblock = PTask::Runtime::AllocateDatablock(port->GetTemplate(), (void *)data, sizeData, NULL);
            channel->Push(pblock);
            pblock->Release();
#endif
        }
    };

    // predeclaration
    template<class ElemType>
    class ComputationNetwork;

    template<class ElemType>
    class ComputationNode;

    // Describes a PTask task.
    // One instance is created for each actual task that will be added to the PTask graph.
    // The descriptors are created first, to support phased assembly of the information 
    // about the tasks and their relationships.
    template<class ElemType>
    class TaskDescriptor
    {
    protected:
        typedef ComputationNode<ElemType>* ComputationNodePtr;

    public:
        TaskDescriptor(
            const ComputationNode<ElemType>* node,
            TaskType taskType,
            size_t input=0
            );

        virtual ~TaskDescriptor();

        const std::string& TaskName() const { return m_taskName; }
        std::string& TaskName() { return m_taskName; }
        bool IsForwardTask() const { return m_taskType == taskEvaluate || m_taskType == taskEvaluateRecurrent; }
        TaskType GetTaskType() const { return m_taskType;}
        const ComputationNodePtr GetNode() const {return m_node;}
        const Task* GetTask() const {return m_task;}

        ParamData<ElemType>* GradientParam(int index=-1, UINT options=paramOptionsInput, ElemType initValue=ElemType(0.0));
        ParamData<ElemType>* FunctionParam(int index=-1, UINT options=paramOptionsOutput);
        ParamData<ElemType>* MatrixParam(const Matrix<ElemType>& matrix, const std::string& name, UINT options=paramOptionsInput);
        ParamData<ElemType>* Param(ParamType paramType, const std::string& name, UINT options=paramOptionsInput, void* data=nullptr);

        void SetFunction(FARPROC function) {m_function = function;}
#ifdef USE_PTASK
        FARPROC GetFunction() {return m_function;}
        void ConfigureInputsAndOutputs(UINT& uidCounter, std::map<const std::string, Port*>& valueNameToProducerPortMap);
        void CreateTask(Graph* graph);

        void CreateChannelsForInputs(
            Graph* graph,
            std::map<const std::string, Port*>& valueNameToProducerPortMap,
            std::map<const std::string, std::vector<PTask::GraphInputChannel*>*>& inputNameToChannelsMap,
            int verbosity);

        void CreateInitializerChannel(
            Graph* graph,
            Port* port,
            Matrix<ElemType>& matrix,
            const std::string& name
            );

        void CreateBackAndInitChannel(Graph* graph, std::map<const std::string, PTask::GraphOutputChannel*>& outputNameToChannelsMap);
        void FindEmptyOutPorts(Graph* graph);

        // GetParamData - return the parameter data in parameter order
        const std::vector<ParamData<ElemType>*>& GetParameters() const {return m_paramData;}
    private:

        Port* TaskDescriptor<ElemType>::CreatePortForTemplate(DatablockTemplate* dt,
            UINT portType, 
            std::string& valueName, 
            UINT portIndex, UINT inoutPort, bool gpuBuffer, 
            UINT& uidCounter,
            std::map<const std::string, Port*>& valueNameToProducerPortMap
            );


        std::vector<const std::string>    m_inputNames;
        std::vector<const std::string>    m_outputNames;

        UINT                            m_numInputPorts;
        Port**                          m_inputPorts;
        UINT                            m_numOutputPorts;
        Port**                          m_outputPorts;

        static DatablockTemplate*       s_descriptorTemplate;
#endif
    private:
        std::vector<ParamData<ElemType>*> m_paramData; // parameter data for CNTK task

        ComputationNodePtr              m_node;
        TaskType                        m_taskType;
        std::string                     m_taskName;

        Task*                           m_task;
        FARPROC                         m_function;
    }; 

    template<class ElemType>
    class PTaskGraphBuilder
    {
    private:
        typedef ComputationNode<ElemType>* ComputationNodePtr;
        typedef TaskDescriptor<ElemType>* TaskDescriptorPtr;
    public:
        PTaskGraphBuilder();
        virtual ~PTaskGraphBuilder(); 

        virtual void BuildFromComputationNetwork(ComputationNetwork<ElemType>* cn);
        void StartPTaskGraph();
        void UpdateParameters(void* sgd, const ElemType learnRatePerSample, const size_t expectedMBSize);

        void PushActualMBSize(const std::list<ComputationNodePtr>& learnableNodes, size_t actualMBSize, CONTROLSIGNAL signal=DBCTLC_NONE);
        void PushData(std::map<std::wstring, Matrix<ElemType>*>& data, CONTROLSIGNAL signal=DBCTLC_NONE);
        void PushMatrix(const Matrix<ElemType>& matrix, Channel* channel, CONTROLSIGNAL signal=DBCTLC_NONE);

        ElemType GetValue(ComputationNodePtr node);
        void GetValue(ComputationNode<ElemType>* node, Matrix<ElemType>& matTo);
#ifdef USE_PTASK

        //static void WINAPI OutputParameter(const ComputationalNode<ElemType>* node, Matrix<ElemType> &functionValues);
        static void __stdcall ApplicationContextCallback(
            APPLICATIONCONTEXTCALLBACKPOINT eCallbackPoint,
            const Datablock * pDatablock,
            void ** ppApplicationContext
            );


    private:
        // Copy constructor, should never be called.
        PTaskGraphBuilder(const PTaskGraphBuilder<ElemType>& deepCopyFrom) {};

        // Assignment operator, should never be called.
        PTaskGraphBuilder<ElemType>& operator=(const PTaskGraphBuilder<ElemType>& deepCopyFrom) {assert(false); return *this; /* NOTE: just doing this to appease the compiler*/};

        // Create descriptors to model PTask tasks that ComputationNodes will be mapped to.
        void CreateTaskDescriptorsForComputationNodes();

        void CreateTaskDescriptorForNode(ComputationNodePtr node, TaskType taskFlag);

        void CreateInputName(std::string inputName);

        // Configure task inputs and outputs.
        void ConfigureTaskInputsAndOutputs()
        {
            if (m_verbosity >= 1) fprintf(stderr, "\nConfiguring task inputs and outputs ...\n");
            for (auto taskIter=m_taskNameToTaskDescriptorMap.begin(); taskIter != m_taskNameToTaskDescriptorMap.end(); taskIter++)
            {                
	            if (m_verbosity >= 2) fprintf(stderr, "  Task %s\n", taskIter->first.c_str());
                TaskDescriptorPtr taskDescriptor = taskIter->second;
                taskDescriptor->ConfigureInputsAndOutputs(m_portUIDCounter, m_valueNameToProducerPortMap);
            }
        }

        // Create actual PTask tasks from task descriptors.
        void CreateTasksFromDescriptors()
        {
            if (m_verbosity >= 1) fprintf(stderr, "\nCreating tasks from descriptors ...\n");
            for (auto taskIter=m_taskNameToTaskDescriptorMap.begin(); taskIter != m_taskNameToTaskDescriptorMap.end(); taskIter++)
            {                
	            if (m_verbosity >= 2) fprintf(stderr, "  Task %s\n", taskIter->first.c_str());
                TaskDescriptorPtr taskDescriptor = taskIter->second;
                taskDescriptor->CreateTask(m_PTaskGraph);
            }
        }

        // Create PTask channels.
        void CreateChannels()
        {
            if (m_verbosity >= 1) fprintf(stderr, "\nCreating PTask channels ...\n");
            for (auto taskIter=m_taskNameToTaskDescriptorMap.begin(); taskIter != m_taskNameToTaskDescriptorMap.end(); taskIter++)
            {                
	            if (m_verbosity >= 2) fprintf(stderr, "  Task %s\n", taskIter->first.c_str());
                TaskDescriptorPtr taskDescriptor = taskIter->second;
                taskDescriptor->CreateChannelsForInputs(m_PTaskGraph, m_valueNameToProducerPortMap, m_inputNameToChannelsMap, m_verbosity);
            }
        }

        void CreateBackAndInitChannels()
        {
            if (m_verbosity >= 1) fprintf(stderr, "\nCreating LearnableParameter extra channels...\n");
            for (auto taskIter=m_taskNameToTaskDescriptorMap.begin(); taskIter != m_taskNameToTaskDescriptorMap.end(); taskIter++)
            {                
	            if (m_verbosity >= 2) fprintf(stderr, "  Task %s\n", taskIter->first.c_str());
                TaskDescriptorPtr taskDescriptor = taskIter->second;
                taskDescriptor->CreateBackAndInitChannel(m_PTaskGraph, m_outputNameToChannelsMap);
            }
        }

        void FindEmptyOutPorts()
        {
            if (m_verbosity >= 1) fprintf(stderr, "\nFinding empty ports to plug...\n");
            for (auto taskIter=m_taskNameToTaskDescriptorMap.begin(); taskIter != m_taskNameToTaskDescriptorMap.end(); taskIter++)
            {                
                TaskDescriptorPtr taskDescriptor = taskIter->second;
                taskDescriptor->FindEmptyOutPorts(m_PTaskGraph);
            }
        }


        void CreatePropogationPath();
        void CreateOutputChannels(const vector<ComputationNodePtr>& nodes);

        // LimitAccelerators - Limit the Accelerators to the ones chosen by the user (or decided by the "bestGPU" algorithm)
        void LimitAccelerators()
        {
            UINT uiIndex = 0;

            // get the devices from here
            std::vector<int>::iterator vi;
            std::vector<int> devices = g_bestGpu->GetDevices(BestGpu::RequeryDevices, bestGpuRequery);
            for(vi=devices.begin(); vi!=devices.end(); vi++)
            {
                if (m_verbosity >= 1)
                    printf("-- CUDA accelerator with CUDA id %d ENABLED for PTask\n", *vi);
                PTask::Runtime::EnableAccelerator(ACCELERATOR_CLASS_CUDA, *vi);
            }
        }


        // Start the PTask graph executing.
        void StartGraph()
        {
            // Output the graph in Graphviz 'dot' format.
            // Obtain Graphviz from http://www.graphviz.org/Download_windows.php
            // Currently using version 2.34. Add C:\Program Files (x86)\Graphviz2.34\bin to PATH.
            // Use:
            //   dot -Tpng C:\temp\PTaskGraph.dot -o PTaskGraph.png 
            // to render as PNG image.
            if (m_verbosity >= 1)
            {
                fprintf(stderr, "Outputting graph to %s in Graphviz 'dot' format ...\n", PTASK_GRAPH_VIZ_FILE);
                fprintf(stderr, "  Convert to .png with: dot -Tpng C:\\temp\\PTaskGraph.dot -o PTaskGraph.png\n");
                fprintf(stderr, "  See PTaskGraphBuilder::StartGraph() for details.\n");
            }
            m_PTaskGraph->WriteDOTFile(PTASK_GRAPH_VIZ_FILE);

            if (m_verbosity >= 1) fprintf(stderr, "Checking graph semantics ...\n");
            Runtime::CheckGraphSemantics(m_PTaskGraph, TRUE, TRUE);

            if (m_verbosity >= 1) fprintf(stderr, "Starting graph ...\n");
            m_PTaskGraph->Run(TRUE); // single threaded for debugging
        }

        bool IsRunning()
        {
            return m_PTaskGraph->IsRunning();
        }
        
        TaskDescriptor<ElemType>* PTaskGraphBuilder<ElemType>::GetPTaskDescriptorOutput(ComputationNodePtr node) const;

        std::vector<ComputationNodePtr>                 m_computationNodes;
        std::map<const std::string, TaskDescriptorPtr>  m_taskNameToTaskDescriptorMap;
        std::map<const std::string, Port*>              m_valueNameToProducerPortMap;
        std::map<const std::string, 
            std::vector<PTask::GraphInputChannel*>*>    m_inputNameToChannelsMap;
        std::map<const std::string, 
            PTask::GraphOutputChannel*>                 m_outputNameToChannelsMap;
        UINT                                            m_portUIDCounter;
        Graph*                                          m_PTaskGraph;

        int                                             m_verbosity;

        // state for PTask nodes
        ComputationNetwork<ElemType>*                   m_cn;
#endif
    };

#ifdef USE_PTASK
    // the Host Task driver
    template <class ElemType>
    static void __stdcall
    HostTaskDriver(LPDEPENDENTCONTEXT depContext);
#endif

}}}
