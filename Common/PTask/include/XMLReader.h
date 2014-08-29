//--------------------------------------------------------------------------------------
// File: XMLReadr.h
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#ifndef _XMLREADER_H_
#define _XMLREADER_H_

#ifdef XMLSUPPORT

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif


#include <vector>
#include <map>
using namespace std;

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

    class XMLReaderException: public std::exception {}; 

    class XMLReader
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   jcurrey, 5/8/2013. </remarks>
        ///
        /// <param name="filename">   The name of the file to read XML from. </param>
        ///-------------------------------------------------------------------------------------------------

        XMLReader(const char * filename);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   jcurrey, 5/8/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~XMLReader();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the graph. </summary>
        ///
        /// <remarks>   jcurrey, originally </remarks>
        ///
        /// <returns>   null if it fails, else the graph. </returns>
        ///-------------------------------------------------------------------------------------------------

        Graph * GetGraph();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets a port. </summary>
        ///
        /// <remarks>   jcurrey, originally </remarks>
        ///
        /// <param name="portUID">  The port UID. </param>
        ///
        /// <returns>   null if it fails, else the port. </returns>
        ///-------------------------------------------------------------------------------------------------

        Port * GetPort(UINT portUID);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Reads a graph. </summary>
        ///
        /// <remarks>   jcurrey, originally. </remarks>
        ///
        /// <param name="pGraph">   [in,out] If non-null, the graph. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL ReadGraph(Graph * pGraph);

        BOOL                ReadStringElement(const char * elementName, std::string& cvalue);
        int                 ReadIntegerElement(const char * elementName);
        UINT                ReadUINTElement(const char * elementName);
        bool                ReadBooleanElement(const char * elementName);


    protected:

        const char*         ReadTextElement(const char * elementName);
        BOOL                ReadTemplates();
        BOOL                ReadKernels();
        BOOL                ReadTasks();
        BOOL                ReadChannels();
        BOOL                ReadActions();
        DatablockTemplate * ReadDatablockTemplate();
        CompiledKernel *    ReadCompiledKernel(int& kernelID);
        Task *              ReadTask();
        Port *              ReadPort();
        Channel *           ReadChannel();
        BOOL                ReadNextNode(XmlNodeType requiredType);
        BOOL                ReadElementStartTag(const char * requiredElementName);      
        BOOL                ReadElementText(const char *& text);
        BOOL                ReadElementEndTag(const char * requiredElementName);
        const wchar_t *     AllocWideStringCopy(const char * str);
        const char *        AllocStringCopy(LPCWSTR strW);
        void                FreeWideString(const wchar_t * str);
        void                FreeString(const char * str);

        IStream *                              m_pInFileStream;
        IXmlReader *                           m_pReader;
                                               
        Graph *                                m_pGraph;
        map<string, DatablockTemplate *>       m_templateMap;
        map<int, CompiledKernel *>             m_kernelMap;
        map<UINT, Port *>                      m_portMap;
        std::set<const wchar_t*>               m_wAllocs;
        std::set<const char*>                  m_cAllocs;
    };

};
#endif
#endif