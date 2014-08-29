///-------------------------------------------------------------------------------------------------
// file:	GeometryEstimator.h
//
// summary:	Declares the geometry estimator class
///-------------------------------------------------------------------------------------------------
#ifndef __GEOMETRY_ESTIMATOR_H__
#define __GEOMETRY_ESTIMATOR_H__

#include "PTaskRuntime.h"
#include <map>

namespace PTask {

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Argument descriptor: provided the peeked value of a datablock and the source port
    ///             from which it was peeked.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef struct _ptask_arg_t {
        Datablock * pBlock;
        Port * pSourcePort;
        Port * pAllocator;
        DatablockTemplate * pPortTemplate;
    } PTASKARGDESC, *PPTASKARGDESC;

	static const int PTGE_DEFAULT_BASIC_GROUP = 256;
	static const int PTGE_DEFAULT_BASIC_GROUP_X = 32;
	static const int PTGE_DEFAULT_BASIC_GROUP_Y = 32;
	static const int PTGE_DEFAULT_BASIC_GROUP_Z = 1;
	static const int PTGE_DEFAULT_ELEMENTS_PER_THREAD = 1;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Geometry estimator callback function prototype. Allows the user is provide a
    ///             custom estimator function.
    ///             </summary>
    ///-------------------------------------------------------------------------------------------------

    typedef void 
    (__stdcall *LPFNGEOMETRYESTIMATOR)(
        UINT nArguments, 
        PTASKARGDESC ** ppArguments, 
        PTASKDIM3 * pBlockDims, 
        PTASKDIM3 * pGridDims,
        int nElementsPerThread,
        int nBasicGroupSizeX,
		int nBasicGroupSizeY,
		int nBasicGroupSizeZ
        );

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Values that represent canonical estimator functions. Most estimators are so
    ///             common that it makes no sense to force the user to code them explicitly. These
    ///             values provide a library of common estimators.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    typedef enum _estimator_fns {

        /// <summary>   No size estimator function has been provided. 
        /// 			</summary>
        NO_SIZE_ESTIMATOR = 0,

        /// <summary>   Estimate the geometry based on the size of the
        /// 			datablock bound to the first port. 
        /// 			</summary>
        BASIC_INPUT_SIZE_ESTIMATOR = 1, //

        /// <summary>   Estimate the geometry based on the max of the
        /// 			record counts over all input datablocks.
        /// 			</summary>
        MAX_INPUT_SIZE_ESTIMATOR = 2, 

        /// <summary>   Estimate the geometry based on the max of the
        /// 			record counts over all output datablocks.
        /// 			</summary>
        MAX_OUTPUT_SIZE_ESTIMATOR = 3,

        /// <summary>   Ports are bound to a particular dimension
        /// 			of the iteration space. This estimator
        /// 			looks for explicit port bindings and assembles
        /// 			the iteration space accordingly. </summary>
        EXPLICIT_DIMENSION_ESTIMATOR = 4,

        /// <summary>   The user commits to provide a callback to
        /// 			estimate the dispatch dimensions. 
        /// 			</summary>
        USER_DEFINED_ESTIMATOR = 5

        // ....

    } GEOMETRYESTIMATORTYPE;

    ///-------------------------------------------------------------------------------------------------
    /// <summary>   Geometry estimator. Functions for estimating dispatch dimensions based on
    ///             dynamically available information.
    ///             </summary>
    ///
    /// <remarks>   crossbac, 5/1/2012. </remarks>
    ///-------------------------------------------------------------------------------------------------

    class GeometryEstimator {

    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Basic Input size geometry estimator. Accepts as input all the datablocks that
        /// 			will be bound to inputs for a given task, but examines only the block bound to
        /// 			parameter 0. This is a legacy function: achtung!
        /// 			</summary>
        ///
        /// <remarks>	crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nArguments">		 	The number of arguments. </param>
        /// <param name="ppArguments">		 	[in] non-null, a vector of input data blocks. </param>
        /// <param name="pBlockDims">		 	[out] non-null, the thread block dimensions. </param>
        /// <param name="pGridDims">		 	[out] non-null, the grid dimensions . </param>
        /// <param name="nElementsPerThread">	(optional) The elements assumed by kernel code to be
        /// 									assigned to each thread. Default is 1. </param>
        /// <param name="nBasicGroupSizeX">  	(optional) size of the basic group. Default is 512. </param>
        /// <param name="nBasicGroupSizeY">  	The basic group size y coordinate. </param>
        /// <param name="nBasicGroupSizeZ">  	The basic group size z coordinate. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        BasicInputSizeGeometryEstimator(
            __in   UINT nArguments, 
            __in   PTask::PTASKARGDESC ** ppArguments, 
            __out  PTask::PTASKDIM3 * pBlockDims, 
            __out PTask::PTASKDIM3 * pGridDims,
            __in   int nElementsPerThread,
            __in   int nBasicGroupSizeX,
            __in   int nBasicGroupSizeY,
            __in   int nBasicGroupSizeZ
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Max Input size geometry estimator. Accepts as input all the datablocks that will
        /// 			be bound to inputs for a given task, and takes the max over all the record counts
        /// 			to find the conservative maximum number of thread blocks that will be required to
        /// 			ensure each input element is processed.
        /// 			</summary>
        ///
        /// <remarks>	crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nArguments">		 	The number of arguments. </param>
        /// <param name="ppArguments">		 	[in] non-null, a vector of input data blocks. </param>
        /// <param name="pBlockDims">		 	[out] non-null, the thread block dimensions. </param>
        /// <param name="pGridDims">		 	[out] non-null, the grid dimensions . </param>
        /// <param name="nElementsPerThread">	(optional) The elements assumed by kernel code to be
        /// 									assigned to each thread. Default is 1. </param>
        /// <param name="nBasicGroupSizeX">  	(optional) size of the basic group. Default is 512. </param>
        /// <param name="nBasicGroupSizeY">  	The basic group size y coordinate. </param>
        /// <param name="nBasicGroupSizeZ">  	The basic group size z coordinate. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        MaxInputSizeGeometryEstimator(
            __in   UINT nArguments, 
            __in   PTask::PTASKARGDESC ** ppArguments, 
            __out  PTask::PTASKDIM3 * pBlockDims, 
            __out PTask::PTASKDIM3 * pGridDims,
            __in   int nElementsPerThread,
            __in   int nBasicGroupSizeX,
            __in   int nBasicGroupSizeY,
            __in   int nBasicGroupSizeZ
			);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Max output size geometry estimator. Accepts as input all the datablocks that will
        /// 			be bound to outputs for a given task, and takes the max over all the record
        /// 			counts to find the conservative maximum number of thread blocks that will be
        /// 			required to ensure each input element is processed. Note that this is a somewhat
        /// 			more subtle task than examining input blocks because output blocks with MetaPorts
        /// 			serving as input allocator will not be allocated yet.
        /// 			</summary>
        ///
        /// <remarks>	crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nArguments">		 	The number of arguments. </param>
        /// <param name="ppArguments">		 	[in] non-null, a vector of input data blocks. </param>
        /// <param name="pBlockDims">		 	[out] non-null, the thread block dimensions. </param>
        /// <param name="pGridDims">		 	[out] non-null, the grid dimensions . </param>
        /// <param name="nElementsPerThread">	(optional) The elements assumed by kernel code to be
        /// 									assigned to each thread. Default is 1. </param>
        /// <param name="nBasicGroupSizeX">  	(optional) size of the basic group. Default is 512. </param>
        /// <param name="nBasicGroupSizeY">  	The basic group size y coordinate. </param>
        /// <param name="nBasicGroupSizeZ">  	The basic group size z coordinate. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        MaxOutputSizeGeometryEstimator(
            __in   UINT nArguments, 
            __in   PTask::PTASKARGDESC ** ppArguments, 
            __out  PTask::PTASKDIM3 * pBlockDims, 
            __out PTask::PTASKDIM3 * pGridDims,
            __in   int nElementsPerThread,
            __in   int nBasicGroupSizeX,
            __in   int nBasicGroupSizeY,
            __in   int nBasicGroupSizeZ
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>	Ports are bound to dimensions of the iteration space such that the datablock size
        /// 			maps directly to one dimension of space. Accept all port/block pairs and use
        /// 			those with an explicit binding to assemble the iteration space.
        /// 			</summary>
        ///
        /// <remarks>	crossbac, 12/20/2011. </remarks>
        ///
        /// <param name="nArguments">		 	The number of arguments. </param>
        /// <param name="ppArguments">		 	[in] non-null, a vector of input data blocks. </param>
        /// <param name="pBlockDims">		 	[out] non-null, the thread block dimensions. </param>
        /// <param name="pGridDims">		 	[out] non-null, the grid dimensions . </param>
        /// <param name="nElementsPerThread">	(optional) The elements assumed by kernel code to be
        /// 									assigned to each thread. Default is 1. </param>
        /// <param name="nBasicGroupSizeX">  	(optional) size of the basic group. Default is 32. </param>
        /// <param name="nBasicGroupSizeY">  	(optional) the basic group size y coordinate. </param>
        /// <param name="nBasicGroupSizeZ">  	(optional) the basic group size z coordinate. </param>
        ///-------------------------------------------------------------------------------------------------

        static void 
        ExplicitDimensionEstimator(
            UINT nArguments, 
            PTask::PTASKARGDESC ** ppArguments, 
            PTask::PTASKDIM3 * pBlockDims, 
            PTask::PTASKDIM3 * pGridDims,
            int nElementsPerThread,
            int nBasicGroupSizeX,
            int nBasicGroupSizeY,
            int nBasicGroupSizeZ
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds peeked blocks from all the ports in the given map to the argument list.
        ///             Helps assemble the argument list input for an estimator.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/1/2012. </remarks>
        ///
        /// <param name="pPortMap">     [in,out] If non-null, the port map. </param>
        /// <param name="ppArgs">       [in,out] If non-null, the arguments. </param>
        /// <param name="nPortIndex">   [in,out] Zero-based index of the n port. </param>
        /// <param name="nMaxToAdd">    (optional) the maximum number of ports to add. -1 means
        ///                             unbounded. </param>
        ///-------------------------------------------------------------------------------------------------
    
        static void
        AddToArgumentList(
            std::map<UINT, Port*>* pPortMap,
            PTASKARGDESC ** ppArgs,
            int &nPortIndex,
            int nMaxToAdd=-1
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Adds peeked blocks from all the ports for all relevant port maps to the argument
        ///             list. Helps assemble the argument list input for an estimator.
        ///             </summary>
        ///
        /// <remarks>   crossbac, 5/1/2012. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the port map. </param>
        /// <param name="pppArgs">  [in,out] If non-null, the ppp arguments. </param>
        ///
        /// <returns>   the number of arguments in the given list. </returns>
        ///-------------------------------------------------------------------------------------------------
    
        static int
        CreateEstimatorArgumentList(
            Task * pTask,
            PTASKARGDESC *** pppArgs
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Estimate task geometry for a cuda task. This implementation is 
        /// 			platform specific because the interface for specifying launch dimensions 
        /// 			is specific to cuda.
        /// 			</summary>
        ///
        /// <remarks>   crossbac, 5/1/2012. </remarks>
        ///
        /// <param name="pTask">    [in,out] If non-null, the task. </param>
        ///-------------------------------------------------------------------------------------------------

        static void
        EstimateCUTaskGeometry(
            Task * pTask
            );
    };


}

#endif // __GEOMETRY_ESTIMATOR_H__