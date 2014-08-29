//--------------------------------------------------------------------------------------
// File: Recorder.h
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#ifndef _RECORDER_H_
#define _RECORDER_H_

#ifndef XMLSUPPORT

//namespace PTask {
//    class BindDescriptorPort { public : BindDescriptorPort(void * pDescribedPort, void * pDescriberPort, int func) {} };
//    class BindControlPort { public : BindControlPort(void * pDescribedPort, void * pDescriberPort, int func) {} };
//    class BindControlPropagationPort { public : BindControlPropagationPort(void * pDescribedPort, void * pDescriberPort) {} };
//    class SetPredicationType { public : SetPredicationType(void * pDescribedPort, int pDescriberPort, int func) {} };
//    class SetComputeGeometry { public : SetComputeGeometry(void * pDescribedPort, int pDescriberPort, int func, int blah) {} };
//    class Recorder { public: static void Record(void * action); };
//}

#define INITRECORDER() 
#define DESTROYRECORDER()
#define RECORDACTION(x,y,z,w) 
#define RECORDACTION2P(x,y,z) 
#define RECORDACTION4P(x,y,z,w,t) 
#else
#define INITRECORDER() Recorder::Initialize()
#define DESTROYRECORDER() Recorder::Destroy()
#define RECORDACTION(x,y,z,w) Recorder::Record(new  PTask::##x((y),(z),(w)))
#define RECORDACTION2P(x,y,z) Recorder::Record(new  PTask::##x((y),(z)))
#define RECORDACTION4P(x,y,z,w,t) Recorder::Record(new  PTask::##x((y),(z),(w),(t)))

#include "XMLWriter.h"
#include "XMLReader.h"
#include "port.h"

namespace PTask {

    class Graph;
    class Task;

    typedef enum _recorded_action_type {

        BINDCONTROLPORT,
        BINDCONTROLPROPAGATIONCHANNEL,
        BINDCONTROLPROPAGATIONPORT,
        BINDDESCRIPTORPORT,
        BINDITERATIONSCOPE,
        SETBLOCKANDGRIDSIZE,
        SETCOMPUTEGEOMETRY,
        SETPREDICATIONTYPE

    } RECORDEDACTIONTYPE;

    class RecordedAction {
    public:
        RecordedAction(RECORDEDACTIONTYPE type, std::string name);
        virtual void Write(XMLWriter * writer)=0;
        virtual void Read(XMLReader * reader)=0;
        virtual void Replay(XMLReader * reader)=0;
        const char * GetName();
        virtual ~RecordedAction() { }

    protected:
        RECORDEDACTIONTYPE m_type;
        std::string        m_name;
    };

    class BindControlPort : public RecordedAction {
    public:
        BindControlPort();
        BindControlPort(
            Port * pController,
            Port * pGatedPort,
            BOOL bInitiallyOpen
        );
        virtual ~BindControlPort() { }

        void Write(XMLWriter * writer);
        void Read(XMLReader * reader);
        void Replay(XMLReader * reader);

    protected:
        UINT   m_controllerPortUID;
        UINT   m_gatedPortUID;
        BOOL   m_initiallyOpen;
    };

    class BindControlPropagationChannel : public RecordedAction {
    public:
        BindControlPropagationChannel();
        BindControlPropagationChannel(
            Port * pInputPort, 
            Channel * pControlledChannel
        );
        virtual ~BindControlPropagationChannel() { }

        void Write(XMLWriter * writer);
        void Read(XMLReader * reader);
        void Replay(XMLReader * reader);

    protected:
        UINT         m_inputPortUID;
        std::string  m_controlledChannelName;
    };

    class BindControlPropagationPort : public RecordedAction {
    public:
        BindControlPropagationPort();
        BindControlPropagationPort(
            Port * pInputPort, 
            Port * pOutputPort
        );
        virtual ~BindControlPropagationPort() { }
        void Write(XMLWriter * writer);
        void Read(XMLReader * reader);
        void Replay(XMLReader * reader);

    protected:
        UINT           m_inputPortUID;
        UINT           m_outputPortUID;
    };

    class BindDescriptorPort : public RecordedAction {
    public:
        BindDescriptorPort();
        BindDescriptorPort(
            Port * pDescribedPort, 
            Port * pDescriberPort,
            DESCRIPTORFUNC func
        );
        virtual ~BindDescriptorPort() { }

        void Write(XMLWriter * writer);
        void Read(XMLReader * reader);
        void Replay(XMLReader * reader);

    protected:
        UINT           m_describedPortUID;
        UINT           m_describerPortUID;
        DESCRIPTORFUNC m_func;
    };

    class BindIterationScope : public RecordedAction {
    public:
        BindIterationScope();
        BindIterationScope(
            Port * pMetaPort, 
            Port * pScopedPort
        );
        virtual ~BindIterationScope() { }

        void Write(XMLWriter * writer);
        void Read(XMLReader * reader);
        void Replay(XMLReader * reader);

    protected:
        UINT           m_metaPortUID;
        UINT           m_scopedPortUID;
    };

    class SetBlockAndGridSize : public RecordedAction {
    public:
        SetBlockAndGridSize();
        SetBlockAndGridSize(
            Task * task,
            PTASKDIM3 grid,
            PTASKDIM3 block
            );
        virtual ~SetBlockAndGridSize() { }

        void Write(XMLWriter * writer);
        void Read(XMLReader * reader);
        void Replay(XMLReader * reader);

    protected:
        std::string  m_taskName;
        PTASKDIM3    m_grid;
        PTASKDIM3    m_block;
    };

    class SetComputeGeometry : public RecordedAction {
    public:
        SetComputeGeometry();
        SetComputeGeometry(
            Task * task,
            int tgx,
            int tgy,
            int tgz);
        virtual ~SetComputeGeometry() { }

        void Write(XMLWriter * writer);
        void Read(XMLReader * reader);
        void Replay(XMLReader * reader);

    protected:
        std::string  m_taskName;
        int          m_tgx;
        int          m_tgy;
        int          m_tgz;    
    };

    class SetPredicationType : public RecordedAction {
    public:
        SetPredicationType();
        SetPredicationType(
            Channel * pChannel,
            CHANNELENDPOINTTYPE eEndpoint, 
            CHANNELPREDICATE eCanonicalPredicator
        );
        virtual ~SetPredicationType() { }

        void Write(XMLWriter * writer);
        void Read(XMLReader * reader);
        void Replay(XMLReader * reader);

    protected:
        std::string  m_channelName;
        int          m_endpointType;
        int          m_canonicalPredicate;
    };

    class Recorder {
    public:
   
        // HACK: Recorder is a singleton for now.
        // TODO: Move to a Recorder per Graph, once can obain handle to Graph instance
        // from all methods which want to record (such as methods on Port and Channel).
        // One possible solution is to move all recordable actions to be methods on Graph.
       static Recorder * Instance();
       static void Record(RecordedAction * action);
       static void Initialize();
       static void Destroy();

       RecordedAction * CreateAction(const char * actionName);
       std::vector<RecordedAction *>* GetRecordedActions();
    
    protected:
       Recorder();
       virtual ~Recorder();
       Recorder(Recorder const&);
       Recorder& operator=(Recorder const&);
       void RecordAction(RecordedAction * action);

       std::vector<RecordedAction *> m_vRecordedActions;
       static Recorder * s_pInstance;
    };

}; // namespace PTask
#endif
#endif