///-------------------------------------------------------------------------------------------------
// file:	Partitioner.h
//
// summary:	Declares the partitioner class
///-------------------------------------------------------------------------------------------------

#ifndef __PARTITIONER_H__
#define __PARTITIONER_H__

#include <Windows.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <crtdbg.h>
#include "primitive_types.h"

namespace PTask {

    class Graph;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Graph partitioner class. Based on Renato et al.'s optimal cut partitioner.
    ///
    ///             Currently, calls out to a .exe. In the future will use a DLL-based version directly.
    ///             Work preparing for the DLL-based version is currently guarded by 
    ///             #ifdef USE_GRAPH_PARTITIONER_DLL
    ///
    /// <remarks>   Crossbac, 12/10/2013. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class Partitioner {
    public: 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   jcurrey, 2/1/2014. </remarks>
        ///
        /// TODO JC params
        ///-------------------------------------------------------------------------------------------------

        Partitioner(
            Graph *         graph, 
            int             numPartitions = 2,
            const char *    workingDir = NULL, 
            const char *    fileNamePrefix = NULL
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   jcurrey 2/1/2014. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Partitioner();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Partition the ptask graph into nPartition. If successful, return true.
        ///
        ///             Currently only 2 partitions are supported.
        ///
        /// <remarks>   jcurrey, 2/1/2014. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------
    
        BOOL Partition();

    protected:
        friend class Graph;

        /// <summary>   The input ptask graph being partitioned. </summary>
        Graph *         m_graph;

        /// <summary>   The number of partitions to divide the graph into. </summary>
        int             m_numPartitions;

        /// <summary>   The directory in which files related to the execution of the partitioner will be written. </summary>
        std::string     m_workingDir;

        /// <summary>   The prefix of the names of the files which will be written. </summary>
        std::string     m_fileNamePrefix;

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Read the partitioner's solution from a file into an array. </summary>
        ///
        /// <remarks>   jcurrey, 2/1/2014. </remarks>
        ///
        /// TODO JC params
        ///-------------------------------------------------------------------------------------------------
        BOOL ReadSolutionFile(
            const char * fileName,
            int expectedNumValues,
            int * values
            );

#ifdef USE_GRAPH_PARTITIONER_DLL
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        Partitioner(Graph * pGraph);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/10/2013. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~Partitioner();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Partition the ptask graph into nPartition. If successful, return true, and set
        ///             nSolutionValue and nSolutionEvaluation, which are (somewhat obscure)
        ///             metrics of the quality of the solution.
        ///             </summary>
        ///
        /// <remarks>   Crossbac, 12/10/2013. </remarks>
        ///
        /// <param name="nPartitions">          The partitions. </param>
        /// <param name="nSolutionValue">       [out] The solution value. </param>
        /// <param name="nSolutionEvaluation">  [out] The solution evaluation. </param>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL Partition(int nPartitions, int& nSolutionValue, int& nSolutionEvaluation);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Assign the partition created by a successful call to Partition to the 
        ///             underlying PTask graph. </summary>
        ///
        /// <remarks>   Crossbac, 12/10/2013. </remarks>
        ///
        /// <returns>   true if it succeeds, false if it fails. </returns>
        ///-------------------------------------------------------------------------------------------------

        BOOL AssignPartition();

    protected:

        /// <summary>   The input ptask graph being partitioned. </summary>
        Graph *     m_pGraph;

        /// <summary>   The solution: an integer-valued partition id per node in m_pGraph </summary>
        int *       m_pSolution;

        /// <summary>   true if the operation was a success, false if it failed. </summary>
        BOOL        m_bSolutionValid;

        /// <summary>   The solution value. </summary>
        int         m_nSolutionValue;

        /// <summary>   The solution evaluation. </summary>
        int         m_nSolutionEvaluation;

        friend class Graph;
#endif // USE_GRAPH_PARTITIONER_DLL

    };
};

#endif  // __PARTITIONER_H__
