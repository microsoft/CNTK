//--------------------------------------------------------------------------------------
// File: datablocktemplate.h
// Maintainer: crossbac@microsoft.com
//--------------------------------------------------------------------------------------
#ifndef _DATABLOCK_TEMPLATE_H_
#define _DATABLOCK_TEMPLATE_H_

#include <stdio.h>
#include <crtdbg.h>
#include <Windows.h>
#include "primitive_types.h"
#include "datablock.h"
#include "ReferenceCounted.h"

using namespace PTask;

///-------------------------------------------------------------------------------------------------
/// <summary>   Values that represent the different points in the lifecycle of a datablock
///             where the application context associated with a datablock can be managed
///             via a callback. </summary>
///
/// <remarks>   jcurrey, 5/1/2014. </remarks>
///-------------------------------------------------------------------------------------------------

typedef enum applicationcontext_callback_point_t {	

        /// <summary> Point at which a datablock is created. </summary>
        CALLBACKPOINT_CREATE,
        
        /// <summary> Point at which a datablock is cloned. </summary>
        CALLBACKPOINT_CLONE,

        /// <summary> Point at which a datablock is destroyed. </summary>
        CALLBACKPOINT_DESTROY

} APPLICATIONCONTEXTCALLBACKPOINT;

///-------------------------------------------------------------------------------------------------
/// <summary>   Function signature of callbacks used to manage the application context associated 
///             with datablocks. set on a per-template basis, via 
///             DatablockTemplate::SetApplicationContextCallback().
///
///             If eCallbackPoint is CALLBACKPOINT_CREATE or CALLBACKPOINT_DESTROY, 
///             ppApplicationContext points to the application context of the datablock being 
///             created or destroyed.
///
///             If eCallbackPoint is CALLBACKPOINT_CLONE, ppApplicationContext points to the 
///             application context of the datablock clone being created. The application context
///             of the datablock being cloned is accessible via pDatablock.
///
///             pDatablock is provided for information only. None of its state should be modified
///             by the callback.
///             </summary>
///
/// <remarks>   jcurrey, 5/1/2014. </remarks>
///
/// <param name="eCallbackPoint">   [in] The point in the datablock's lifecycle at which the callback was called. </param>
/// <param name="pDatablock">   [in] The datablock being created, cloned or destroyed. </param>
/// <param name="ppApplicationContext">   [inout] The application context to be managed. </param>
///-------------------------------------------------------------------------------------------------

typedef void (__stdcall *LPFNAPPLICATIONCONTEXTCALLBACK)(
    __in    APPLICATIONCONTEXTCALLBACKPOINT eCallbackPoint,
    __in    const Datablock * pDatablock,
    __inout void ** ppApplicationContext
    );

namespace PTask {

    class DatablockTemplate : public ReferenceCounted
    {
    public:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
        /// <param name="uiElementStride">      [in] The element stride in bytes. </param>
        /// <param name="uiElementsX">          [in] Number of elements in X dimension. </param>
        /// <param name="uiElementsY">          [in] Number of elements in Y dimension. </param>
        /// <param name="uiElementsZ">          [in] Number of elements in Z dimension. </param>
        /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
        /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate(
            __in char *       lpszTemplateName, 
            __in unsigned int uiElementStride, 
            __in unsigned int uiElementsX, 
            __in unsigned int uiElementsY, 
            __in unsigned int uiElementsZ,
            __in bool         bIsRecordStream,
            __in bool         bIsByteAddressable
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
        /// <param name="uiElementStride">      [in] The element stride in bytes. </param>
        /// <param name="uiElementsX">          [in] Number of elements in X dimension. </param>
        /// <param name="uiElementsY">          [in] Number of elements in Y dimension. </param>
        /// <param name="uiElementsZ">          [in] Number of elements in Z dimension. </param>
        /// <param name="uiPitch">              [in] The row pitch. </param>
        /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
        /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate(
            __in char *       lpszTemplateName, 
            __in unsigned int uiElementStride, 
            __in unsigned int uiElementsX, 
            __in unsigned int uiElementsY, 
            __in unsigned int uiElementsZ,
            __in unsigned int uiPitch,
            __in bool         bIsRecordStream,
            __in bool         bIsByteAddressable
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpszTemplateName">     [in] If non-null, name of the template. </param>
        /// <param name="pBufferDims">          [in] The element stride in bytes. </param>
        /// <param name="uiNumBufferDims">      [in] Number of elements in X dimension. </param>
        /// <param name="bIsRecordStream">      [in] true if this object is record stream. </param>
        /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate(
            __in char *             lpszTemplateName, 
            __in BUFFERDIMENSIONS * pBufferDims, 
            __in unsigned int       uiNumBufferDims, 
            __in bool               bIsRecordStream,
            __in bool               bIsByteAddressable
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Constructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpszTemplateName">         [in] If non-null, name of the template. </param>
        /// <param name="uiElementStride">          [in] The element stride in bytes. </param>
        /// <param name="describedParameterType">   [in] Type of the described parameter. </param>
        ///-------------------------------------------------------------------------------------------------

        DatablockTemplate(
            __in char * lpszTemplateName, 
            __in unsigned int uiElementStride, 
            __in PTASK_PARM_TYPE describedParameterType
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Destructor. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///-------------------------------------------------------------------------------------------------

        virtual ~DatablockTemplate();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the stride. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        unsigned int GetStride(UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of elements in X. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        unsigned int GetXElementCount(UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of elements in Y. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        unsigned int GetYElementCount(UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of elements in Z. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        unsigned int GetZElementCount(UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of elements in Z. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        unsigned int GetTotalElementCount(UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of elements in Z. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        unsigned int GetDimensionElementCount(UINT uiDim, UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the pitch. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiChannelIndex">   (optional) zero-based index of the channel. </param>
        ///
        /// <returns>   The pitch. </returns>
        ///-------------------------------------------------------------------------------------------------

        unsigned int GetPitch(UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets buffer dimensions. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiChannelIndex">   (optional) zero-based index of the channel. </param>
        ///
        /// <returns>   The buffer dimensions. </returns>
        ///-------------------------------------------------------------------------------------------------

        BUFFERDIMENSIONS GetBufferDimensions(UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets buffer dimensions. </summary>
        ///
        /// <remarks>   Crossbac, 2/18/2013. </remarks>
        ///
        /// <param name="uiChannelIndex">   (optional) zero-based index of the channel. </param>
        ///
        /// <returns>   The buffer dimensions. </returns>
        ///-------------------------------------------------------------------------------------------------

        void SetBufferDimensions(BUFFERDIMENSIONS &dims, UINT uiChannelIndex=DBDATA_IDX);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the datablock byte count. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The datablock byte count. </returns>
        ///-------------------------------------------------------------------------------------------------

        unsigned int GetDatablockByteCount(UINT nChannelIndex=0);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is byte-addressable. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if raw, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual bool IsByteAddressable();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object is variable dimensioned. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if variable dimensioned, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual bool IsVariableDimensioned();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets whether the template describes byte addressable blocks. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="bIsByteAddressable">   [in] true if this object is byte addressable. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        SetByteAddressable(
            __in bool bIsByteAddressable
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this template describes blocks that
        /// 			comprise a record stream. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if the template indicates a record stream. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL DescribesRecordStream();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Return true if this template describes blocks that
        /// 			are used as scalar parameter in kernel functions. 
        /// 			</summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   true if the template describes scalar parameter blocks. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL DescribesScalarParameter();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the parameter base type. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The parameter base type. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual PTASK_PARM_TYPE GetParameterBaseType();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Sets the default value. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <param name="lpvInitData">      [in] If non-null, information describing the lpv initialise. </param>
        /// <param name="cbData">           [in] The data. </param>
        /// <param name="nRecordCount">     [in] Number of records. </param>
        /// <param name="bExplicitlyEmpty"> [in] True if this initializer describes an explicitly empty
        ///                                 initial value (0-length) We track this explicitly because
        ///                                 creating resources based on such initial values that can
        ///                                 actually be bound to device-side execution parameters
        ///                                 necessitates the creation of non-zero-size buffers, whose
        ///                                 logical length is still 0. Hence, we must decouple the
        ///                                 tracking of the "empty" property from whether the init buffer
        ///                                 is null or has no length in general. A null initializer does
        ///                                 not necessarily mean no initializer has been set! </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void 
        SetInitialValue(
            __in void * lpvInitData, 
            __in UINT cbData,
            __in UINT nRecordCount,
            __in BOOL bExplicitlyEmpty=FALSE
            );

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the initial value size. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The initial value size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetInitialValueSizeBytes();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the number of elements in the initial value. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   The initial value size. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetInitialValueElements();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the initial value. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the initial value. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual const void * GetInitialValue();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has an initial value. </summary>
        ///
        /// <remarks>   crossbac, 6/15/2012. </remarks>
        ///
        /// <returns>   true if initial value, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL HasInitialValue();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has an initial value that can be recreated easily
        /// 			using a memset (rather than a memcpy). The object is memsettable if it has
        /// 			an initial value whose size is less than 4 bytes, or whose initial value
        /// 			is identical for all elements when the value is interpreted as either a 4-byte
        /// 			int or an unsigned char. 
        /// 			</summary>
        ///
        /// <remarks>   crossbac, 6/15/2012. </remarks>
        ///
        /// <returns>   true if initial value, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsInitialValueMemsettable(UINT szPrimitiveSize=0);

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Query if this object has an initial value that can be recreated easily
        /// 			using a memset (rather than a memcpy), restricted to 8 bit objects.
        /// 			</summary>
        ///
        /// <remarks>   crossbac, 6/15/2012. </remarks>
        ///
        /// <returns>   true if initial value, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsInitialValueMemsettableD8();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the initial value memset stride. </summary>
        ///
        /// <remarks>   crossbac, 7/6/2012. </remarks>
        ///
        /// <returns>   The initial value memset stride. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual UINT GetInitialValueMemsetStride();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Queries if the initial value for this template is empty. </summary>
        ///
        /// <remarks>   crossbac, 6/15/2012. </remarks>
        ///
        /// <returns>   true if an initial value is empty, false if not. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual BOOL IsInitialValueEmpty(); 

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Gets the type. </summary>
        ///
        /// <remarks>   Crossbac, 12/23/2011. </remarks>
        ///
        /// <returns>   null if it fails, else the type. </returns>
        ///-------------------------------------------------------------------------------------------------

        virtual char * GetTemplateName();

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Set the application context callback function associated with this 
        ///             datablock template. </summary>
        ///
        /// <remarks>   jcurrey, 5/1/2014. </remarks>
        ///
        /// <param name="pCallback"> [in] The callback function to associate with this template. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual void SetApplicationContextCallback(LPFNAPPLICATIONCONTEXTCALLBACK pCallback);
    
        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Get the application context callback function associated with this 
        ///             datablock template. </summary>
        ///
        /// <remarks>   jcurrey, 5/1/2014. </remarks>
        ///
        /// <returns>   The callback function associated with this template. </param>
        ///-------------------------------------------------------------------------------------------------

        virtual LPFNAPPLICATIONCONTEXTCALLBACK GetApplicationContextCallback();
    
    protected:

        ///-------------------------------------------------------------------------------------------------
        /// <summary>   Default initialize. </summary>
        ///
        /// <remarks>   crossbac, 7/9/2012. </remarks>
        ///
        /// <param name="lpszTemplateName"> [in,out] If non-null, name of the template. </param>
        ///-------------------------------------------------------------------------------------------------

        void DefaultInitialize(char * lpszTemplateName);
    
        /// <summary> true if this template describes a 
        /// 		  record stream 
        /// 		  </summary>
        bool          m_bRecordStream;
        
        /// <summary> true if this template describes 
        /// 		  byte-addressable datablocks
        /// 		  </summary>
        bool          m_bByteAddressable;
        
        /// <summary> true if this template describes blocks
        /// 		  that are used as scalar parameters in
        /// 		  kernel invocations </summary>
        bool          m_bScalarParameter;
        
        /// <summary> The parameter base type</summary>
        PTASK_PARM_TYPE     m_bParameterBaseType;
        
        /// <summary> The name of datablock template,
        /// 		  user-supplied (in a hopefully
        /// 		  descriptive way) 
        /// 		  </summary>
        char *		  m_lpszTemplateName;

#if 0
        /// <summary> The stride in bytes of a single
        /// 		  element in a block created with
        /// 		  this template. 
        /// 		  </summary>
        unsigned int  m_uiStride;

        /// <summary>   The vui channel dimensions. </summary>
        unsigned int*   m_pChannelDimensions[NUM_DATABLOCK_CHANNELS];

        /// <summary>   Sizes of the three dimensions of elements in blocks created with this template.
        ///             </summary>
        unsigned int    m_vuiDataDimensions[MAX_DATABLOCK_DIMENSIONS];

        /// <summary>   The vui meta dimensions. </summary>
        unsigned int    m_vuiMetaDimensions[MAX_DATABLOCK_DIMENSIONS];

        /// <summary>   The vui template data dimensions. </summary>
        unsigned int    m_vuiTemplateDataDimensions[MAX_DATABLOCK_DIMENSIONS];
#endif
        
        /// <summary>   The channel dimensions, per channel type. </summary>
        BUFFERDIMENSIONS m_vChannelDimensions[NUM_DATABLOCK_CHANNELS];
        
        /// <summary>   An (optional) initial value. </summary>
        void *          m_lpvInitialValue;

        /// <summary>   Size of the initial value buffer if such a buffer is extant. </summary>
        UINT            m_cbInitialValue;

        /// <summary>   Number of records in the initial value. Generally speaking this
        /// 			value should be the same as m_cbInitialValue/stride, but we 
        /// 			insist on this redundancy to enable sanity checking. </summary>
        UINT            m_nInitialRecordCount;

        /// <summary>   true if the initial value is explicitly empty, meaning that a null
        /// 			m_lpvInitialValue pointer or 0-valued m_cbInitialValue does not indicate
        /// 			the absence of an initializer for this template. 
        /// 			</summary>
        BOOL            m_bExplicitlyEmptyInitialValue;

        /// <summary>   true if we have already checked whether this template
        /// 			has an initial value that can be created with a memset 
        /// 			call (rather than a memcpy). </summary>
        BOOL            m_bMemsetCheckComplete;
        
        /// <summary>   true if the initial value can be created with memset. 
        /// 			Valid only if m_bMemsetCheckComplete is true. 
        /// 			</summary>
        BOOL            m_bMemsettableInitialValue;

        /// <summary>   true if the initial value can be created with memset. 
        /// 			Valid only if m_bMemsetCheckComplete is true. 
        /// 			</summary>
        BOOL            m_bMemsettableInitialValueD8;

        /// <summary>   The memsettable initial value (byte-granularity). </summary>
        unsigned char   m_ucMemsettableInitialValueD8;

        /// <summary>   The memset initial value stride. </summary>
        UINT            m_bMemsetInitialValueStride;

        /// <summary>   The application context callback. </summary>
        LPFNAPPLICATIONCONTEXTCALLBACK m_pApplicationContextCallback;

    };

};
#endif