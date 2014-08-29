//--------------------------------------------------------------------------------------
// File: XMLWriter.h
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#ifndef _XMLWRITER_H_
#define _XMLWRITER_H_

#ifdef XMLSUPPORT

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include <vector>
#include <map>

#include <ole2.h>
#include <xmllite.h>
#include <stdio.h>
#include <shlwapi.h>

/*#include "graph.h"
#include "datablock.h"
#include "datablocktemplate.h"
#include "CompiledKernel.h"
#include "primitive_types.h"
*/

#include "primitive_types.h"
#include "PTaskRuntime.h"
#include "channel.h"

namespace PTask {

    class XMLWriterException: public std::exception {}; 

    class XMLWriter
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   jcurrey, 5/5/2013. </remarks>
        ///
        /// <param name="filename">   The name of the file to write XML to. </param>
        ///-------------------------------------------------------------------------------------------------

        XMLWriter(const char * filename);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   jcurrey, 5/5/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~XMLWriter();

        void WriteElementStartTag(const char * elementName);
        void WriteElementText(const char * text);
        void WriteElementEndTag();
        void WriteComment(const char * comment);
        void WriteEndDocument();

        void WriteElement(const char * elementName, const char * text);
        void WriteElement(const char * elementName, int elementValue);
        void WriteElement(const char * elementName, unsigned int elementValue);
        void WriteElement(const char * elementName, bool elementValue);

        void WriteGraph(Graph * pGraph);
        void WriteDatablockTemplate(DatablockTemplate * pTemplate);
        void WriteCompiledKernel(CompiledKernel * pCompiledKernel, int kernelID);
        void WriteTask(Task * pTask, int kernelID);
        void WritePorts(std::map<UINT, Port*>* pPorts);
        void WritePort(Port * pPort);
        void WriteControlPropagationInfo(Port * pPort);
        void WriteChannel(Channel * pChannel);
        void WriteChannelEndpointPredication(Channel * pChannel, CHANNELENDPOINTTYPE eEndpoint);

    protected:
        const wchar_t * ToWChar(const char * str);

        IStream *    m_pOutFileStream;
        IXmlWriter * m_pWriter;
    };

};
#endif
#endif