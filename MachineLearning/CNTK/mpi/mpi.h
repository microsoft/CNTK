/*
 *  (C) 2001 by Argonne National Laboratory.
 *  (C) 2009 by Microsoft Corporation.
 *      See COPYRIGHT in the SDK directory.
 */

#ifndef MPI_INCLUDED
#define MPI_INCLUDED

#if defined(__cplusplus)
extern "C" {
#endif


#ifndef MSMPI_VER
#define MSMPI_VER 0x100
#endif


/*---------------------------------------------------------------------------*/
/* SAL ANNOTATIONS                                                           */
/*---------------------------------------------------------------------------*/
/*
 * Define SAL annotations if they aren't defined yet.  Note that if we define
 * them, we undefine them at the end of this file so that future inclusion of
 * sal.h doesn't cause errors.
 */
#ifndef _In_
#define MSMPI_DEFINED_SAL
#define _In_
#define _In_z_
#define _In_opt_
#define _In_count_( x )
#define _In_bytecount_( x )
#define _In_opt_count_( x )
#define _Out_
#define _Out_cap_( x )
#define _Out_cap_post_count_( x, y )
#define _Out_bytecap_( x )
#define _Out_z_cap_( x )
#define _Out_z_cap_post_count_( x, y )
#define _Out_cap_post_part_( x, y )
#define _Out_opt_
#define _Out_opt_cap_( x )
#define _Post_z_
#define _Inout_
#define _Inout_count_( x )
#endif

/*---------------------------------------------------------------------------*/
/* MSMPI Calling convention                                                  */
/*---------------------------------------------------------------------------*/

#define MPIAPI __stdcall


/*---------------------------------------------------------------------------*/
/* MPI ERROR CLASS                                                           */
/*---------------------------------------------------------------------------*/

#define MPI_SUCCESS          0      /* Successful return code */

#define MPI_ERR_BUFFER       1      /* Invalid buffer pointer */
#define MPI_ERR_COUNT        2      /* Invalid count argument */
#define MPI_ERR_TYPE         3      /* Invalid datatype argument */
#define MPI_ERR_TAG          4      /* Invalid tag argument */
#define MPI_ERR_COMM         5      /* Invalid communicator */
#define MPI_ERR_RANK         6      /* Invalid rank */
#define MPI_ERR_ROOT         7      /* Invalid root */
#define MPI_ERR_GROUP        8      /* Invalid group */
#define MPI_ERR_OP           9      /* Invalid operation */
#define MPI_ERR_TOPOLOGY    10      /* Invalid topology */
#define MPI_ERR_DIMS        11      /* Invalid dimension argument */
#define MPI_ERR_ARG         12      /* Invalid argument */
#define MPI_ERR_UNKNOWN     13      /* Unknown error */
#define MPI_ERR_TRUNCATE    14      /* Message truncated on receive */
#define MPI_ERR_OTHER       15      /* Other error; use Error_string */
#define MPI_ERR_INTERN      16      /* Internal error code */
#define MPI_ERR_IN_STATUS   17      /* Error code is in status */
#define MPI_ERR_PENDING     18      /* Pending request */
#define MPI_ERR_REQUEST     19      /* Invalid request (handle) */
#define MPI_ERR_ACCESS      20      /* Premission denied */
#define MPI_ERR_AMODE       21      /* Error related to amode passed to MPI_File_open */
#define MPI_ERR_BAD_FILE    22      /* Invalid file name (e.g., path name too long) */
#define MPI_ERR_CONVERSION  23      /* Error in user data conversion function */
#define MPI_ERR_DUP_DATAREP 24      /* Data representation identifier already registered */
#define MPI_ERR_FILE_EXISTS 25      /* File exists */
#define MPI_ERR_FILE_IN_USE 26      /* File operation could not be completed, file in use */
#define MPI_ERR_FILE        27      /* Invalid file handle */
#define MPI_ERR_INFO        28      /* Invalid info argument */
#define MPI_ERR_INFO_KEY    29      /* Key longer than MPI_MAX_INFO_KEY */
#define MPI_ERR_INFO_VALUE  30      /* Value longer than MPI_MAX_INFO_VAL */
#define MPI_ERR_INFO_NOKEY  31      /* Invalid key passed to MPI_Info_delete */
#define MPI_ERR_IO          32      /* Other I/O error */
#define MPI_ERR_NAME        33      /* Invalid service name in MPI_Lookup_name */
#define MPI_ERR_NO_MEM      34      /* Alloc_mem could not allocate memory */
#define MPI_ERR_NOT_SAME    35      /* Collective argument/sequence not the same on all processes */
#define MPI_ERR_NO_SPACE    36      /* Not enough space */
#define MPI_ERR_NO_SUCH_FILE 37     /* File does not exist */
#define MPI_ERR_PORT        38      /* Invalid port name in MPI_comm_connect*/
#define MPI_ERR_QUOTA       39      /* Quota exceeded */
#define MPI_ERR_READ_ONLY   40      /* Read-only file or file system */
#define MPI_ERR_SERVICE     41      /* Invalid service name in MPI_Unpublish_name */
#define MPI_ERR_SPAWN       42      /* Error in spawning processes */
#define MPI_ERR_UNSUPPORTED_DATAREP   43  /* Unsupported dararep in MPI_File_set_view */
#define MPI_ERR_UNSUPPORTED_OPERATION 44  /* Unsupported operation on file */
#define MPI_ERR_WIN         45      /* Invalid win argument */
#define MPI_ERR_BASE        46      /* Invalid base passed to MPI_Free_mem */
#define MPI_ERR_LOCKTYPE    47      /* Invalid locktype argument */
#define MPI_ERR_KEYVAL      48      /* Invalid keyval  */
#define MPI_ERR_RMA_CONFLICT 49     /* Conflicting accesses to window */
#define MPI_ERR_RMA_SYNC    50      /* Wrong synchronization of RMA calls */
#define MPI_ERR_SIZE        51      /* Invalid size argument */
#define MPI_ERR_DISP        52      /* Invalid disp argument */
#define MPI_ERR_ASSERT      53      /* Invalid assert argument */

#define MPI_ERR_LASTCODE    0x3fffffff    /* Last valid error code for a predefined error class */

#define MPICH_ERR_LAST_CLASS 53


/*---------------------------------------------------------------------------*/
/* MPI Basic integer types                                                   */
/*---------------------------------------------------------------------------*/

/* Address size integer */
#ifdef _WIN64
typedef __int64 MPI_Aint;
#else
typedef int MPI_Aint;
#endif

/* Fortran INTEGER */
typedef int MPI_Fint;

/* File offset */
typedef __int64 MPI_Offset;


/*---------------------------------------------------------------------------*/
/* MPI_Datatype                                                              */
/*---------------------------------------------------------------------------*/

typedef int MPI_Datatype;
#define MPI_DATATYPE_NULL           ((MPI_Datatype)0x0c000000)

#define MPI_CHAR                    ((MPI_Datatype)0x4c000101)
#define MPI_UNSIGNED_CHAR           ((MPI_Datatype)0x4c000102)
#define MPI_SHORT                   ((MPI_Datatype)0x4c000203)
#define MPI_UNSIGNED_SHORT          ((MPI_Datatype)0x4c000204)
#define MPI_INT                     ((MPI_Datatype)0x4c000405)
#define MPI_UNSIGNED                ((MPI_Datatype)0x4c000406)
#define MPI_LONG                    ((MPI_Datatype)0x4c000407)
#define MPI_UNSIGNED_LONG           ((MPI_Datatype)0x4c000408)
#define MPI_LONG_LONG_INT           ((MPI_Datatype)0x4c000809)
#define MPI_LONG_LONG               MPI_LONG_LONG_INT
#define MPI_FLOAT                   ((MPI_Datatype)0x4c00040a)
#define MPI_DOUBLE                  ((MPI_Datatype)0x4c00080b)
#define MPI_LONG_DOUBLE             ((MPI_Datatype)0x4c00080c)
#define MPI_BYTE                    ((MPI_Datatype)0x4c00010d)
#define MPI_WCHAR                   ((MPI_Datatype)0x4c00020e)

#define MPI_PACKED                  ((MPI_Datatype)0x4c00010f)
#define MPI_LB                      ((MPI_Datatype)0x4c000010)
#define MPI_UB                      ((MPI_Datatype)0x4c000011)

#define MPI_C_COMPLEX               ((MPI_Datatype)0x4c000812)
#define MPI_C_FLOAT_COMPLEX         ((MPI_Datatype)0x4c000813)
#define MPI_C_DOUBLE_COMPLEX        ((MPI_Datatype)0x4c001614)
#define MPI_C_LONG_DOUBLE_COMPLEX   ((MPI_Datatype)0x4c001615)

#define MPI_2INT                    ((MPI_Datatype)0x4c000816)
#define MPI_C_BOOL                  ((MPI_Datatype)0x4c000117)
#define MPI_SIGNED_CHAR             ((MPI_Datatype)0x4c000118)
#define MPI_UNSIGNED_LONG_LONG      ((MPI_Datatype)0x4c000819)

/* Fortran types */
#define MPI_CHARACTER               ((MPI_Datatype)0x4c00011a)
#define MPI_INTEGER                 ((MPI_Datatype)0x4c00041b)
#define MPI_REAL                    ((MPI_Datatype)0x4c00041c)
#define MPI_LOGICAL                 ((MPI_Datatype)0x4c00041d)
#define MPI_COMPLEX                 ((MPI_Datatype)0x4c00081e)
#define MPI_DOUBLE_PRECISION        ((MPI_Datatype)0x4c00081f)
#define MPI_2INTEGER                ((MPI_Datatype)0x4c000820)
#define MPI_2REAL                   ((MPI_Datatype)0x4c000821)
#define MPI_DOUBLE_COMPLEX          ((MPI_Datatype)0x4c001022)
#define MPI_2DOUBLE_PRECISION       ((MPI_Datatype)0x4c001023)
#define MPI_2COMPLEX                ((MPI_Datatype)0x4c001024)
#define MPI_2DOUBLE_COMPLEX         ((MPI_Datatype)0x4c002025)

/* Size-specific types (see MPI 2.2, 16.2.5) */
#define MPI_REAL2                   MPI_DATATYPE_NULL
#define MPI_REAL4                   ((MPI_Datatype)0x4c000427)
#define MPI_COMPLEX8                ((MPI_Datatype)0x4c000828)
#define MPI_REAL8                   ((MPI_Datatype)0x4c000829)
#define MPI_COMPLEX16               ((MPI_Datatype)0x4c00102a)
#define MPI_REAL16                  MPI_DATATYPE_NULL
#define MPI_COMPLEX32               MPI_DATATYPE_NULL
#define MPI_INTEGER1                ((MPI_Datatype)0x4c00012d)
#define MPI_COMPLEX4                MPI_DATATYPE_NULL
#define MPI_INTEGER2                ((MPI_Datatype)0x4c00022f)
#define MPI_INTEGER4                ((MPI_Datatype)0x4c000430)
#define MPI_INTEGER8                ((MPI_Datatype)0x4c000831)
#define MPI_INTEGER16               MPI_DATATYPE_NULL
#define MPI_INT8_T                  ((MPI_Datatype)0x4c000133)
#define MPI_INT16_T                 ((MPI_Datatype)0x4c000234)
#define MPI_INT32_T                 ((MPI_Datatype)0x4c000435)
#define MPI_INT64_T                 ((MPI_Datatype)0x4c000836)
#define MPI_UINT8_T                 ((MPI_Datatype)0x4c000137)
#define MPI_UINT16_T                ((MPI_Datatype)0x4c000238)
#define MPI_UINT32_T                ((MPI_Datatype)0x4c000439)
#define MPI_UINT64_T                ((MPI_Datatype)0x4c00083a)

#ifdef _WIN64
#define MPI_AINT                    ((MPI_Datatype)0x4c00083b)
#else
#define MPI_AINT                    ((MPI_Datatype)0x4c00043b)
#endif
#define MPI_OFFSET                  ((MPI_Datatype)0x4c00083c)

/*
 * The layouts for the types MPI_DOUBLE_INT etc. are
 *
 *      struct { double a; int b; }
 */
#define MPI_FLOAT_INT               ((MPI_Datatype)0x8c000000)
#define MPI_DOUBLE_INT              ((MPI_Datatype)0x8c000001)
#define MPI_LONG_INT                ((MPI_Datatype)0x8c000002)
#define MPI_SHORT_INT               ((MPI_Datatype)0x8c000003)
#define MPI_LONG_DOUBLE_INT         ((MPI_Datatype)0x8c000004)


/*---------------------------------------------------------------------------*/
/* MPI_Comm                                                                  */
/*---------------------------------------------------------------------------*/

typedef int MPI_Comm;
#define MPI_COMM_NULL  ((MPI_Comm)0x04000000)

#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)
#define MPI_COMM_SELF  ((MPI_Comm)0x44000001)


/*---------------------------------------------------------------------------*/
/* MPI_Win                                                                   */
/*---------------------------------------------------------------------------*/

typedef int MPI_Win;
#define MPI_WIN_NULL ((MPI_Win)0x20000000)


/*---------------------------------------------------------------------------*/
/* MPI_File                                                                  */
/*---------------------------------------------------------------------------*/

typedef struct ADIOI_FileD* MPI_File;
#define MPI_FILE_NULL ((MPI_File)0)


/*---------------------------------------------------------------------------*/
/* MPI_Op                                                                    */
/*---------------------------------------------------------------------------*/

typedef int MPI_Op;
#define MPI_OP_NULL ((MPI_Op)0x18000000)

#define MPI_MAX     ((MPI_Op)0x58000001)
#define MPI_MIN     ((MPI_Op)0x58000002)
#define MPI_SUM     ((MPI_Op)0x58000003)
#define MPI_PROD    ((MPI_Op)0x58000004)
#define MPI_LAND    ((MPI_Op)0x58000005)
#define MPI_BAND    ((MPI_Op)0x58000006)
#define MPI_LOR     ((MPI_Op)0x58000007)
#define MPI_BOR     ((MPI_Op)0x58000008)
#define MPI_LXOR    ((MPI_Op)0x58000009)
#define MPI_BXOR    ((MPI_Op)0x5800000a)
#define MPI_MINLOC  ((MPI_Op)0x5800000b)
#define MPI_MAXLOC  ((MPI_Op)0x5800000c)
#define MPI_REPLACE ((MPI_Op)0x5800000d)


/*---------------------------------------------------------------------------*/
/* MPI_Info                                                                  */
/*---------------------------------------------------------------------------*/

typedef int MPI_Info;
#define MPI_INFO_NULL         ((MPI_Info)0x1c000000)


/*---------------------------------------------------------------------------*/
/* MPI_Request                                                               */
/*---------------------------------------------------------------------------*/

typedef int MPI_Request;
#define MPI_REQUEST_NULL ((MPI_Request)0x2c000000)


/*---------------------------------------------------------------------------*/
/* MPI_Group                                                                 */
/*---------------------------------------------------------------------------*/

typedef int MPI_Group;
#define MPI_GROUP_NULL  ((MPI_Group)0x08000000)

#define MPI_GROUP_EMPTY ((MPI_Group)0x48000000)


/*---------------------------------------------------------------------------*/
/* MPI_Errhandler                                                            */
/*---------------------------------------------------------------------------*/

typedef int MPI_Errhandler;
#define MPI_ERRHANDLER_NULL  ((MPI_Errhandler)0x14000000)

#define MPI_ERRORS_ARE_FATAL ((MPI_Errhandler)0x54000000)
#define MPI_ERRORS_RETURN    ((MPI_Errhandler)0x54000001)

/*---------------------------------------------------------------------------*/
/* MPI_Status                                                                */
/*---------------------------------------------------------------------------*/

typedef struct MPI_Status
{
    int count;
    int cancelled;
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;

} MPI_Status;

#define MPI_STATUS_IGNORE ((MPI_Status*)(MPI_Aint)1)
#define MPI_STATUSES_IGNORE ((MPI_Status*)(MPI_Aint)1)


/*---------------------------------------------------------------------------*/
/* MISC CONSTANTS                                                            */
/*---------------------------------------------------------------------------*/

/* Used in: Count, Index, Rank, Color, Toplogy, Precision, Exponent range  */
#define MPI_UNDEFINED   (-32766)

/* Used in: Rank */
#define MPI_PROC_NULL   (-1)
#define MPI_ANY_SOURCE  (-2)
#define MPI_ROOT        (-3)

/* Used in: Tag */
#define MPI_ANY_TAG     (-1)

/* Used for: Buffer address */
#define MPI_BOTTOM      ((void*)0)


/*---------------------------------------------------------------------------*/
/* Chapter 3: Point-to-Point Communication                                   */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------*/
/* Section 3.2: Blocking Communication         */
/*---------------------------------------------*/

int
MPIAPI
MPI_Send(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Send(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Recv(
    _Out_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Recv(
    _Out_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Get_count(
    _In_ MPI_Status* status,
    MPI_Datatype datatype,
    _Out_ int* count
    );
int
MPIAPI
PMPI_Get_count(
    _In_ MPI_Status* status,
    MPI_Datatype datatype,
    _Out_ int* count
    );


/*---------------------------------------------*/
/* Section 3.4: Communication Modes            */
/*---------------------------------------------*/

int
MPIAPI
MPI_Bsend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Bsend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Ssend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Ssend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Rsend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Rsend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm
    );


/*---------------------------------------------*/
/* Section 3.6: Buffer Allocation              */
/*---------------------------------------------*/

/* Upper bound on bsend overhead for each message */
#define MSMPI_BSEND_OVERHEAD_V1   95
#define MSMPI_BSEND_OVERHEAD_V2   MSMPI_BSEND_OVERHEAD_V1

#if MSMPI_VER > 0x300
#  define MPI_BSEND_OVERHEAD  MSMPI_Get_bsend_overhead()
#else
#  define MPI_BSEND_OVERHEAD  MSMPI_BSEND_OVERHEAD_V1
#endif

int
MPIAPI
MPI_Buffer_attach(
    _In_ void* buffer,
    int size
    );
int
MPIAPI
PMPI_Buffer_attach(
    _In_ void* buffer,
    int size
    );

int
MPIAPI
MPI_Buffer_detach(
    _Out_ void* buffer_addr,
    _Out_ int* size
    );
int
MPIAPI
PMPI_Buffer_detach(
    _Out_ void* buffer_addr,
    _Out_ int* size
    );


/*---------------------------------------------*/
/* Section 3.7: Nonblocking Communication      */
/*---------------------------------------------*/

int
MPIAPI
MPI_Isend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Isend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Ibsend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Ibsend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Issend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Issend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Irsend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Irsend(
    _In_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Irecv(
    _Out_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Irecv(
    _Out_opt_ void* buf,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );


/*---------------------------------------------*/
/* Section 3.7.3: Communication Completion     */
/*---------------------------------------------*/

int
MPIAPI
MPI_Wait(
    _Inout_ MPI_Request* request,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Wait(
    _Inout_ MPI_Request* request,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Test(
    _Inout_ MPI_Request* request,
    _Out_ int* flag,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Test(
    _Inout_ MPI_Request* request,
    _Out_ int* flag,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Request_free(
    _Inout_ MPI_Request* request
    );
int
MPIAPI
PMPI_Request_free(
    _Inout_ MPI_Request* request
    );


/*---------------------------------------------*/
/* Section 3.7.5: Multiple Completions         */
/*---------------------------------------------*/

int
MPIAPI
MPI_Waitany(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests,
    _Out_ int* index,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Waitany(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests,
    _Out_ int* index,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Testany(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests,
    _Out_ int* index,
    _Out_ int* flag,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Testany(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests,
    _Out_ int* index,
    _Out_ int* flag,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Waitall(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests,
    _Out_cap_(count) MPI_Status* array_of_statuses
    );
int
MPIAPI
PMPI_Waitall(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests,
    _Out_cap_(count) MPI_Status* array_of_statuses
    );

int
MPIAPI
MPI_Testall(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests,
    _Out_ int* flag,
    _Out_cap_(count) MPI_Status* array_of_statuses
    );
int
MPIAPI
PMPI_Testall(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests,
    _Out_ int* flag,
    _Out_cap_(count) MPI_Status* array_of_statuses
    );

int
MPIAPI
MPI_Waitsome(
    int incount,
    _Inout_count_(incount) MPI_Request* array_of_requests,
    _Out_ int* outcount,
    _Out_cap_post_count_(incount,*outcount) int* array_of_indices,
    _Out_cap_post_count_(incount,*outcount) MPI_Status* array_of_statuses
    );
int
MPIAPI
PMPI_Waitsome(
    int incount,
    _Inout_count_(incount) MPI_Request* array_of_requests,
    _Out_ int* outcount,
    _Out_cap_post_count_(incount,*outcount) int* array_of_indices,
    _Out_cap_post_count_(incount,*outcount) MPI_Status* array_of_statuses
    );

int
MPIAPI
MPI_Testsome(
    int incount,
    _Inout_count_(incount) MPI_Request* array_of_requests,
    _Out_ int* outcount,
    _Out_cap_post_count_(incount,*outcount) int* array_of_indices,
    _Out_cap_post_count_(incount,*outcount) MPI_Status* array_of_statuses
    );
int
MPIAPI
PMPI_Testsome(
    int incount,
    _Inout_count_(incount) MPI_Request* array_of_requests,
    _Out_ int* outcount,
    _Out_cap_post_count_(incount,*outcount) int* array_of_indices,
    _Out_cap_post_count_(incount,*outcount) MPI_Status* array_of_statuses
    );


/*---------------------------------------------*/
/* Section 3.7.6: Test of status               */
/*---------------------------------------------*/

int
MPIAPI
MPI_Request_get_status(
    MPI_Request request,
    _Out_ int* flag,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Request_get_status(
    MPI_Request request,
    _Out_ int* flag,
    _Out_ MPI_Status* status
    );


/*---------------------------------------------*/
/* Section 3.8: Probe and Cancel               */
/*---------------------------------------------*/

int
MPIAPI
MPI_Iprobe(
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ int* flag,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Iprobe(
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ int* flag,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Probe(
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Probe(
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Cancel(
    _In_ MPI_Request* request
    );
int
MPIAPI
PMPI_Cancel(
    _In_ MPI_Request* request
    );

int
MPIAPI
MPI_Test_cancelled(
    _In_ MPI_Status* status,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Test_cancelled(
    _In_ MPI_Status* request,
    _Out_ int* flag
    );


/*---------------------------------------------*/
/* Section 3.9: Persistent Communication       */
/*---------------------------------------------*/

int
MPIAPI
MPI_Send_init(
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Send_init(
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Bsend_init(
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Bsend_init(
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Ssend_init(
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Ssend_init(
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Rsend_init(
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Rsend_init(
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Recv_init(
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Recv_init(
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm comm,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Start(
    _Inout_ MPI_Request* request
    );
int
MPIAPI
PMPI_Start(
    _Inout_ MPI_Request* request
    );

int
MPIAPI
MPI_Startall(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests
    );
int
MPIAPI
PMPI_Startall(
    int count,
    _Inout_count_(count) MPI_Request* array_of_requests
    );


/*---------------------------------------------*/
/* Section 3.10: Send-Recv                     */
/*---------------------------------------------*/

int
MPIAPI
MPI_Sendrecv(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    int dest,
    int sendtag,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int source,
    int recvtag,
    MPI_Comm comm,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Sendrecv(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    int dest,
    int sendtag,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int source,
    int recvtag,
    MPI_Comm comm,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Sendrecv_replace(
    _Inout_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int sendtag,
    int source,
    int recvtag,
    MPI_Comm comm,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Sendrecv_replace(
    _Inout_ void* buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int sendtag,
    int source,
    int recvtag,
    MPI_Comm comm,
    _Out_ MPI_Status* status
    );


/*---------------------------------------------------------------------------*/
/* Chapter 4: Datatypes                                                      */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------*/
/* Section 4.1: Derived Datatypes              */
/*---------------------------------------------*/

int
MPIAPI
MPI_Type_contiguous(
    int count,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_contiguous(
    int count,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );

int
MPIAPI
MPI_Type_vector(
    int count,
    int blocklength,
    int stride,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_vector(
    int count,
    int blocklength,
    int stride,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );

int
MPIAPI
MPI_Type_create_hvector(
    int count,
    int blocklength,
    MPI_Aint stride,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_hvector(
    int count,
    int blocklength,
    MPI_Aint stride,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );

int
MPIAPI
MPI_Type_indexed(
    int count,
    _In_count_(count) int* array_of_blocklengths,
    _In_count_(count) int* array_of_displacements,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_indexed(
    int count,
    _In_count_(count) int* array_of_blocklengths,
    _In_count_(count) int* array_of_displacements,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );

int
MPIAPI
MPI_Type_create_hindexed(
    int count,
    _In_count_(count) int array_of_blocklengths[],
    _In_count_(count) MPI_Aint array_of_displacements[],
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_hindexed(
    int count,
    _In_count_(count) int array_of_blocklengths[],
    _In_opt_count_(count) MPI_Aint array_of_displacements[],
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );

int
MPIAPI
MPI_Type_create_indexed_block(
    int count,
    int blocklength,
    _In_count_(count) int array_of_displacements[],
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_indexed_block(
    int count,
    int blocklength,
    _In_count_(count) int array_of_displacements[],
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );

int
MPIAPI
MPI_Type_create_struct(
    int count,
    _In_count_(count) int array_of_blocklengths[],
    _In_count_(count) MPI_Aint array_of_displacements[],
    _In_count_(count) MPI_Datatype array_of_types[],
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_struct(
    int count,
    _In_count_(count) int array_of_blocklengths[],
    _In_opt_count_(count) MPI_Aint array_of_displacements[],
    _In_count_(count) MPI_Datatype array_of_types[],
    _Out_ MPI_Datatype* newtype
    );


#define MPI_ORDER_C         56
#define MPI_ORDER_FORTRAN   57

int
MPIAPI
MPI_Type_create_subarray(
    int ndims,
    _In_count_(ndims) int array_of_sizes[],
    _In_count_(ndims) int array_of_subsizes[],
    _In_count_(ndims) int array_of_starts[],
    int order,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_subarray(
    int ndims,
    _In_count_(ndims) int array_of_sizes[],
    _In_count_(ndims) int array_of_subsizes[],
    _In_count_(ndims) int array_of_starts[],
    int order,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );


#define MPI_DISTRIBUTE_BLOCK         121
#define MPI_DISTRIBUTE_CYCLIC        122
#define MPI_DISTRIBUTE_NONE          123
#define MPI_DISTRIBUTE_DFLT_DARG (-49767)

int
MPIAPI
MPI_Type_create_darray(
    int size,
    int rank,
    int ndims,
    _In_count_(ndims) int array_of_gszies[],
    _In_count_(ndims) int array_of_distribs[],
    _In_count_(ndims) int array_of_dargs[],
    _In_count_(ndims) int array_of_psizes[],
    int order,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_darray(
    int size,
    int rank,
    int ndims,
    _In_count_(ndims) int array_of_gszies[],
    _In_count_(ndims) int array_of_distribs[],
    _In_count_(ndims) int array_of_dargs[],
    _In_count_(ndims) int array_of_psizes[],
    int order,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );


/*---------------------------------------------*/
/* Section 4.1.5: Datatype Address and Size    */
/*---------------------------------------------*/

int
MPIAPI
MPI_Get_address(
    _In_ void* location,
    _Out_ MPI_Aint* address
    );
int
MPIAPI
PMPI_Get_address(
    _In_ void* location,
    _Out_ MPI_Aint* address
    );

int
MPIAPI
MPI_Type_size(
    MPI_Datatype datatype,
    _Out_ int* size
    );
int
MPIAPI
PMPI_Type_size(
    MPI_Datatype datatype,
    _Out_ int* size
    );


/*---------------------------------------------*/
/* Section 4.1.7: Datatype Extent and Bounds   */
/*---------------------------------------------*/

int
MPIAPI
MPI_Type_get_extent(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* lb,
    _Out_ MPI_Aint* extent
    );
int
MPIAPI
PMPI_Type_get_extent(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* lb,
    _Out_ MPI_Aint* extent
    );

int
MPIAPI
MPI_Type_create_resized(
    MPI_Datatype oldtype,
    MPI_Aint lb,
    MPI_Aint extent,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_resized(
    MPI_Datatype oldtype,
    MPI_Aint lb,
    MPI_Aint extent,
    _Out_ MPI_Datatype* newtype
    );


/*---------------------------------------------*/
/* Section 4.1.8: Datatype True Extent         */
/*---------------------------------------------*/

int
MPIAPI
MPI_Type_get_true_extent(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* true_lb,
    _Out_ MPI_Aint* true_extent
    );
int
MPIAPI
PMPI_Type_get_true_extent(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* true_lb,
    _Out_ MPI_Aint* true_extent
    );


/*---------------------------------------------*/
/* Section 4.1.9: Datatype Commit and Free     */
/*---------------------------------------------*/

int
MPIAPI
MPI_Type_commit(
    _In_ MPI_Datatype* datatype
    );
int
MPIAPI
PMPI_Type_commit(
    _In_ MPI_Datatype* datatype
    );

int
MPIAPI
MPI_Type_free(
    _Inout_ MPI_Datatype* datatype
    );
int
MPIAPI
PMPI_Type_free(
    _Inout_ MPI_Datatype* datatype
    );


/*---------------------------------------------*/
/* Section 4.1.10: Datatype Duplication        */
/*---------------------------------------------*/

int
MPIAPI
MPI_Type_dup(
    MPI_Datatype type,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_dup(
    MPI_Datatype type,
    _Out_ MPI_Datatype* newtype
    );


/*---------------------------------------------*/
/* Section 4.1.11: Datatype and Communication  */
/*---------------------------------------------*/

int
MPIAPI
MPI_Get_elements(
    _In_ MPI_Status* status,
    MPI_Datatype datatype,
    _Out_ int* count
    );
int
MPIAPI
PMPI_Get_elements(
    _In_ MPI_Status* status,
    MPI_Datatype datatype,
    _Out_ int* count
    );


/*---------------------------------------------*/
/* Section 4.1.13: Decoding a Datatype         */
/*---------------------------------------------*/

/* Datatype combiners result */
enum
{
    MPI_COMBINER_NAMED            = 1,
    MPI_COMBINER_DUP              = 2,
    MPI_COMBINER_CONTIGUOUS       = 3,
    MPI_COMBINER_VECTOR           = 4,
    MPI_COMBINER_HVECTOR_INTEGER  = 5,
    MPI_COMBINER_HVECTOR          = 6,
    MPI_COMBINER_INDEXED          = 7,
    MPI_COMBINER_HINDEXED_INTEGER = 8,
    MPI_COMBINER_HINDEXED         = 9,
    MPI_COMBINER_INDEXED_BLOCK    = 10,
    MPI_COMBINER_STRUCT_INTEGER   = 11,
    MPI_COMBINER_STRUCT           = 12,
    MPI_COMBINER_SUBARRAY         = 13,
    MPI_COMBINER_DARRAY           = 14,
    MPI_COMBINER_F90_REAL         = 15,
    MPI_COMBINER_F90_COMPLEX      = 16,
    MPI_COMBINER_F90_INTEGER      = 17,
    MPI_COMBINER_RESIZED          = 18
};

int
MPIAPI
MPI_Type_get_envelope(
    MPI_Datatype datatype,
    _Out_ int* num_integers,
    _Out_ int* num_addresses,
    _Out_ int* num_datatypes,
    _Out_ int* combiner
    );
int
MPIAPI
PMPI_Type_get_envelope(
    MPI_Datatype datatype,
    _Out_ int* num_integers,
    _Out_ int* num_addresses,
    _Out_ int* num_datatypes,
    _Out_ int* combiner
    );

int
MPIAPI
MPI_Type_get_contents(
    MPI_Datatype datatype,
    int max_integers,
    int max_addresses,
    int max_datatypes,
    _Out_cap_(max_integers) int array_of_integers[],
    _Out_cap_(max_addresses) MPI_Aint array_of_addresses[],
    _Out_cap_(max_datatypes) MPI_Datatype array_of_datatypes[]
    );
int
MPIAPI
PMPI_Type_get_contents(
    MPI_Datatype datatype,
    int max_integers,
    int max_addresses,
    int max_datatypes,
    _Out_cap_(max_integers) int array_of_integers[],
    _Out_cap_(max_addresses) MPI_Aint array_of_addresses[],
    _Out_cap_(max_datatypes) MPI_Datatype array_of_datatypes[]
    );


/*---------------------------------------------*/
/* Section 4.2: Datatype Pack and Unpack       */
/*---------------------------------------------*/

int
MPIAPI
MPI_Pack(
    _In_ void* inbuf,
    int incount,
    MPI_Datatype datatype,
    _Out_bytecap_(outsize) void* outbuf,
    int outsize,
    _Inout_ int* position,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Pack(
    _In_ void* inbuf,
    int incount,
    MPI_Datatype datatype,
    _Out_bytecap_(outsize) void* outbuf,
    int outsize,
    _Inout_ int* position,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Unpack(
    _In_bytecount_(insize) void* inbuf,
    int insize,
    _Inout_ int* position,
    _Out_ void* outbuf,
    int outcount,
    MPI_Datatype datatype,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Unpack(
    _In_bytecount_(insize) void* inbuf,
    int insize,
    _Inout_ int* position,
    _Out_ void* outbuf,
    int outcount,
    MPI_Datatype datatype,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Pack_size(
    int incount,
    MPI_Datatype datatype,
    MPI_Comm comm,
    _Out_ int* size
    );
int
MPIAPI
PMPI_Pack_size(
    int incount,
    MPI_Datatype datatype,
    MPI_Comm comm,
    _Out_ int* size
    );


/*---------------------------------------------*/
/* Section 4.3: Canonical Pack and Unpack      */
/*---------------------------------------------*/

int
MPIAPI
MPI_Pack_external(
    _In_z_ char* datarep,
    _In_ void* inbuf,
    int incount,
    MPI_Datatype datatype,
    _Out_bytecap_(outsize) void* outbuf,
    MPI_Aint outsize,
    _Inout_ MPI_Aint* position
    );
int
MPIAPI
PMPI_Pack_external(
    _In_z_ char* datarep,
    _In_ void* inbuf,
    int incount,
    MPI_Datatype datatype,
    _Out_bytecap_(outsize) void* outbuf,
    MPI_Aint outsize,
    _Inout_ MPI_Aint* position
    );

int
MPIAPI
MPI_Unpack_external(
    _In_z_ char* datarep,
    _In_bytecount_(insize) void* inbuf,
    MPI_Aint insize,
    _Inout_ MPI_Aint* position,
    _Out_ void* outbuf,
    int outcount,
    MPI_Datatype datatype
    );
int
MPIAPI
PMPI_Unpack_external(
    _In_z_ char* datarep,
    _In_bytecount_(insize) void* inbuf,
    MPI_Aint insize,
    _Inout_ MPI_Aint* position,
    _Out_ void* outbuf,
    int outcount,
    MPI_Datatype datatype
    );

int
MPIAPI
MPI_Pack_external_size(
    _In_z_ char* datarep,
    int incount,
    MPI_Datatype datatype,
    _Out_ MPI_Aint* size
    );
int
MPIAPI
PMPI_Pack_external_size(
    _In_z_ char* datarep,
    int incount,
    MPI_Datatype datatype,
    _Out_ MPI_Aint* size
    );


/*---------------------------------------------------------------------------*/
/* Chapter 5: Collective Communication                                       */
/*---------------------------------------------------------------------------*/

#define MPI_IN_PLACE ((void*)(MPI_Aint)-1)

/*---------------------------------------------*/
/* Section 5.3: Barrier Synchronization        */
/*---------------------------------------------*/

int
MPIAPI
MPI_Barrier(
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Barrier(
    MPI_Comm comm
    );


/*---------------------------------------------*/
/* Section 5.4: Broadcast                      */
/*---------------------------------------------*/

int
MPIAPI
MPI_Bcast(
    _Inout_ void* buffer,
    int count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Bcast(
    _Inout_ void* buffer,
    int count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm comm
    );


/*---------------------------------------------*/
/* Section 5.5: Gather                         */
/*---------------------------------------------*/

int
MPIAPI
MPI_Gather(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_opt_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Gather(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_opt_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Gatherv(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_opt_ void* recvbuf,
    _In_opt_ int* recvcounts,
    _In_opt_ int* displs,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Gatherv(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_opt_ void* recvbuf,
    _In_opt_ int* recvcounts,
    _In_opt_ int* displs,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
    );


/*---------------------------------------------*/
/* Section 5.6: Scatter                        */
/*---------------------------------------------*/

int
MPIAPI
MPI_Scatter(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Scatter(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Scatterv(
    _In_ void* sendbuf,
    _In_ int* sendcounts,
    _In_ int* displs,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Scatterv(
    _In_ void* sendbuf,
    _In_ int* sendcounts,
    _In_ int* displs,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    int root,
    MPI_Comm comm
    );


/*---------------------------------------------*/
/* Section 5.6: Gather-to-all                  */
/*---------------------------------------------*/

int
MPIAPI
MPI_Allgather(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Allgather(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Allgatherv(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    _In_ int* recvcounts,
    _In_ int* displs,
    MPI_Datatype recvtype,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Allgatherv(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    _In_ int* recvcounts,
    _In_ int* displs,
    MPI_Datatype recvtype,
    MPI_Comm comm
    );


/*---------------------------------------------*/
/* Section 5.6: All-to-All Scatter/Gather      */
/*---------------------------------------------*/

int
MPIAPI
MPI_Alltoall(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Alltoall(
    _In_ void* sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Alltoallv(
    _In_ void* sendbuf,
    _In_ int* sendcounts,
    _In_ int* sdispls,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    _In_ int* recvcounts,
    _In_ int* rdispls,
    MPI_Datatype recvtype,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Alltoallv(
    _In_ void* sendbuf,
    _In_ int* sendcounts,
    _In_ int* sdispls,
    MPI_Datatype sendtype,
    _Out_ void* recvbuf,
    _In_ int* recvcounts,
    _In_ int* rdispls,
    MPI_Datatype recvtype,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Alltoallw(
    _In_ void* sendbuf,
    _In_ int sendcounts[],
    _In_ int sdispls[],
    _In_ MPI_Datatype sendtypes[],
    _Out_ void* recvbuf,
    _In_ int recvcounts[],
    _In_ int rdispls[],
    _In_ MPI_Datatype recvtypes[],
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Alltoallw(
    _In_ void* sendbuf,
    _In_ int sendcounts[],
    _In_ int sdispls[],
    _In_ MPI_Datatype sendtypes[],
    _Out_ void* recvbuf,
    _In_ int recvcounts[],
    _In_ int rdispls[],
    _In_ MPI_Datatype recvtypes[],
    MPI_Comm comm
    );


/*---------------------------------------------*/
/* Section 5.9: Global Reduction Operations    */
/*---------------------------------------------*/

typedef
void
(MPIAPI MPI_User_function)(
    _In_count_(*len) void* invec,
    _Inout_ void* inoutvec,
    _In_ int* len,
    _In_ MPI_Datatype* datatype
    );

int
MPIAPI
MPI_Op_create(
    _In_ MPI_User_function* function,
    int commute,
    _Out_ MPI_Op* op
    );
int
MPIAPI
PMPI_Op_create(
    _In_ MPI_User_function* function,
    int commute,
    _Out_ MPI_Op* op
    );

int
MPIAPI
MPI_Op_free(
    _Inout_ MPI_Op* op
    );
int
MPIAPI
PMPI_Op_free(
    _Inout_ MPI_Op* op
    );

int
MPIAPI
MPI_Reduce(
    _In_ void* sendbuf,
    _Out_opt_ void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Reduce(
    _In_ void* sendbuf,
    _Out_opt_ void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Allreduce(
    _In_ void* sendbuf,
    _Out_ void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Allreduce(
    _In_ void* sendbuf,
    _Out_ void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Reduce_local(
    _In_ void *inbuf,
    _Inout_ void *inoutbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op
    );
int
MPIAPI
PMPI_Reduce_local(
    _In_ void *inbuf,
    _Inout_ void *inoutbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op
    );

/*---------------------------------------------*/
/* Section 5.10: Reduce-Scatter                */
/*---------------------------------------------*/

int
MPIAPI
MPI_Reduce_scatter(
    _In_ void* sendbuf,
    _Out_ void* recvbuf,
    _In_ int* recvcounts,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Reduce_scatter(
    _In_ void* sendbuf,
    _Out_ void* recvbuf,
    _In_ int* recvcounts,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
    );


/*---------------------------------------------*/
/* Section 5.11: Scan                          */
/*---------------------------------------------*/

int
MPIAPI
MPI_Scan(
    _In_ void* sendbuf,
    _Out_ void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Scan(
    _In_ void* sendbuf,
    _Out_ void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
    );

int
MPIAPI
MPI_Exscan(
    _In_ void* sendbuf,
    _Out_ void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
    );
int
MPIAPI
PMPI_Exscan(
    _In_ void* sendbuf,
    _Out_ void* recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm
    );


/*---------------------------------------------------------------------------*/
/* Chapter 6: Groups, Contexts, Communicators, and Caching                   */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------*/
/* Section 6.3: Group Management               */
/*---------------------------------------------*/

int
MPIAPI
MPI_Group_size(
    MPI_Group group,
    _Out_ int* size
    );
int
MPIAPI
PMPI_Group_size(
    MPI_Group group,
    _Out_ int* size
    );

int
MPIAPI
MPI_Group_rank(
    MPI_Group group,
    _Out_ int* rank
    );
int
MPIAPI
PMPI_Group_rank(
    MPI_Group group,
    _Out_ int* rank
    );

int
MPIAPI
MPI_Group_translate_ranks(
    MPI_Group group1,
    int n,
    _In_count_(n) int* ranks1,
    MPI_Group group2,
    _Out_ int* ranks2
    );

int
MPIAPI
PMPI_Group_translate_ranks(
    MPI_Group group1,
    int n,
    _In_count_(n) int* ranks1,
    MPI_Group group2,
    _Out_ int* ranks2
    );

/* Results of the compare operations */
#define MPI_IDENT       0
#define MPI_CONGRUENT   1
#define MPI_SIMILAR     2
#define MPI_UNEQUAL     3

int
MPIAPI
MPI_Group_compare(
    MPI_Group group1,
    MPI_Group group2,
    _Out_ int* result
    );
int
MPIAPI
PMPI_Group_compare(
    MPI_Group group1,
    MPI_Group group2,
    _Out_ int* result
    );

int
MPIAPI
MPI_Comm_group(
    MPI_Comm comm,
    _Out_ MPI_Group* group
    );
int
MPIAPI
PMPI_Comm_group(
    MPI_Comm comm,
    _Out_ MPI_Group* group
    );

int
MPIAPI
MPI_Group_union(
    MPI_Group group1,
    MPI_Group group2,
    _Out_ MPI_Group* newgroup
    );
int
MPIAPI
PMPI_Group_union(
    MPI_Group group1,
    MPI_Group group2,
    _Out_ MPI_Group* newgroup
    );

int
MPIAPI
MPI_Group_intersection(
    MPI_Group group1,
    MPI_Group group2,
    _Out_ MPI_Group* newgroup
    );
int
MPIAPI
PMPI_Group_intersection(
    MPI_Group group1,
    MPI_Group group2,
    _Out_ MPI_Group* newgroup
    );

int
MPIAPI
MPI_Group_difference(
    MPI_Group group1,
    MPI_Group group2,
    _Out_ MPI_Group* newgroup
    );
int
MPIAPI
PMPI_Group_difference(
    MPI_Group group1,
    MPI_Group group2,
    _Out_ MPI_Group* newgroup
    );

int
MPIAPI
MPI_Group_incl(
    MPI_Group group,
    int n,
    _In_count_(n) int* ranks,
    _Out_ MPI_Group* newgroup
    );
int
MPIAPI
PMPI_Group_incl(
    MPI_Group group,
    int n,
    _In_count_(n) int* ranks,
    _Out_ MPI_Group* newgroup
    );

int
MPIAPI
MPI_Group_excl(
    MPI_Group group,
    int n,
    _In_count_(n) int* ranks,
    _Out_ MPI_Group* newgroup
    );
int
MPIAPI
PMPI_Group_excl(
    MPI_Group group,
    int n,
    _In_count_(n) int* ranks,
    _Out_ MPI_Group* newgroup
    );

int
MPIAPI
MPI_Group_range_incl(
    MPI_Group group,
    int n,
    _In_count_(n) int ranges[][3],
    _Out_ MPI_Group* newgroup
    );
int
MPIAPI
PMPI_Group_range_incl(
    MPI_Group group,
    int n,
    _In_count_(n) int ranges[][3],
    _Out_ MPI_Group* newgroup
    );

int
MPIAPI
MPI_Group_range_excl(
    MPI_Group group,
    int n,
    _In_count_(n) int ranges[][3],
    _Out_ MPI_Group* newgroup
    );
int
MPIAPI
PMPI_Group_range_excl(
    MPI_Group group,
    int n,
    _In_count_(n) int ranges[][3],
    _Out_ MPI_Group* newgroup
    );

int
MPIAPI
MPI_Group_free(
    _Inout_ MPI_Group* group
    );
int
MPIAPI
PMPI_Group_free(
    _Inout_ MPI_Group* group
    );


/*---------------------------------------------*/
/* Section 6.4: Communicator Management        */
/*---------------------------------------------*/

int
MPIAPI
MPI_Comm_size(
    MPI_Comm comm,
    _Out_ int* size
    );
int
MPIAPI
PMPI_Comm_size(
    MPI_Comm comm,
    _Out_ int* size
    );

int
MPIAPI
MPI_Comm_rank(
    MPI_Comm comm,
    _Out_ int* rank
    );
int
MPIAPI
PMPI_Comm_rank(
    MPI_Comm comm,
    _Out_ int* rank
    );

int
MPIAPI
MPI_Comm_compare(
    MPI_Comm comm1,
    MPI_Comm comm2,
    _Out_ int* result
    );
int
MPIAPI
PMPI_Comm_compare(
    MPI_Comm comm1,
    MPI_Comm comm2,
    _Out_ int* result
    );

int
MPIAPI
MPI_Comm_dup(
    MPI_Comm comm,
    _Out_ MPI_Comm* newcomm
    );
int
MPIAPI
PMPI_Comm_dup(
    MPI_Comm comm,
    _Out_ MPI_Comm* newcomm
    );

int
MPIAPI
MPI_Comm_create(
    MPI_Comm comm,
    MPI_Group group,
    _Out_ MPI_Comm* newcomm
    );
int
MPIAPI
PMPI_Comm_create(
    MPI_Comm comm,
    MPI_Group group,
    _Out_ MPI_Comm* newcomm
    );

int
MPIAPI
MPI_Comm_split(
    MPI_Comm comm,
    int color,
    int key,
    _Out_ MPI_Comm* newcomm
    );
int
MPIAPI
PMPI_Comm_split(
    MPI_Comm comm,
    int color,
    int key,
    _Out_ MPI_Comm* newcomm
    );

int
MPIAPI
MPI_Comm_free(
    _Inout_ MPI_Comm* comm
    );
int
MPIAPI
PMPI_Comm_free(
    _Inout_ MPI_Comm* comm
    );


/*---------------------------------------------*/
/* Section 6.6: Inter-Communication            */
/*---------------------------------------------*/

int
MPIAPI
MPI_Comm_test_inter(
    MPI_Comm comm,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Comm_test_inter(
    MPI_Comm comm,
    _Out_ int* flag
    );

int
MPIAPI
MPI_Comm_remote_size(
    MPI_Comm comm,
    _Out_ int* size
    );
int
MPIAPI
PMPI_Comm_remote_size(
    MPI_Comm comm,
    _Out_ int* size
    );

int
MPIAPI
MPI_Comm_remote_group(
    MPI_Comm comm,
    _Out_ MPI_Group* group
    );
int
MPIAPI
PMPI_Comm_remote_group(
    MPI_Comm comm,
    _Out_ MPI_Group* group
    );

int
MPIAPI
MPI_Intercomm_create(
    MPI_Comm local_comm,
    int local_leader,
    MPI_Comm peer_comm,
    int remote_leader,
    int tag,
    _Out_ MPI_Comm* newintercomm
    );
int
MPIAPI
PMPI_Intercomm_create(
    MPI_Comm local_comm,
    int local_leader,
    MPI_Comm peer_comm,
    int remote_leader,
    int tag,
    _Out_ MPI_Comm* newintercomm
    );

int
MPIAPI
MPI_Intercomm_merge(
    MPI_Comm intercomm,
    int high,
    _Out_ MPI_Comm* newintracomm
    );
int
MPIAPI
PMPI_Intercomm_merge(
    MPI_Comm intercomm,
    int high,
    _Out_ MPI_Comm* newintracomm
    );


/*---------------------------------------------*/
/* Section 6.7: Caching                        */
/*---------------------------------------------*/

#define MPI_KEYVAL_INVALID  0x24000000

typedef
int
(MPIAPI MPI_Comm_copy_attr_function)(
    MPI_Comm oldcomm,
    int comm_keyval,
    _In_opt_ void* extra_state,
    _In_ void* attribute_val_in,
    _Out_ void* attribute_val_out,
    _Out_ int* flag
    );

typedef
int
(MPIAPI MPI_Comm_delete_attr_function)(
    MPI_Comm comm,
    int comm_keyval,
    _In_ void* attribute_val,
    _In_opt_ void* extra_state
    );

#define MPI_COMM_NULL_COPY_FN ((MPI_Comm_copy_attr_function*)0)
#define MPI_COMM_NULL_DELETE_FN ((MPI_Comm_delete_attr_function*)0)
#define MPI_COMM_DUP_FN ((MPI_Comm_copy_attr_function*)MPIR_Dup_fn)

int
MPIAPI
MPI_Comm_create_keyval(
    _In_opt_ MPI_Comm_copy_attr_function* comm_copy_attr_fn,
    _In_opt_ MPI_Comm_delete_attr_function* comm_delete_attr_fn,
    _Out_ int* comm_keyval,
    _In_opt_ void* extra_state
    );
int
MPIAPI
PMPI_Comm_create_keyval(
    _In_opt_ MPI_Comm_copy_attr_function* comm_copy_attr_fn,
    _In_opt_ MPI_Comm_delete_attr_function* comm_delete_attr_fn,
    _Out_ int* comm_keyval,
    _In_opt_ void* extra_state
    );

int
MPIAPI
MPI_Comm_free_keyval(
    _Inout_ int* comm_keyval
    );
int
MPIAPI
PMPI_Comm_free_keyval(
    _Inout_ int* comm_keyval
    );

int
MPIAPI
MPI_Comm_set_attr(
    MPI_Comm comm,
    int comm_keyval,
    _In_ void* attribute_val
    );
int
MPIAPI
PMPI_Comm_set_attr(
    MPI_Comm comm,
    int comm_keyval,
    _In_ void* attribute_val
    );


/* Predefined comm attribute key values */
/* C Versions (return pointer to value),
   Fortran Versions (return integer value).

   DO NOT CHANGE THESE.  The values encode:
   builtin kind (0x1 in bit 30-31)
   Keyval object (0x9 in bits 26-29)
   for communicator (0x1 in bits 22-25)

   Fortran versions of the attributes are formed by adding one to
   the C version.
 */
#define MPI_TAG_UB          0x64400001
#define MPI_HOST            0x64400003
#define MPI_IO              0x64400005
#define MPI_WTIME_IS_GLOBAL 0x64400007
#define MPI_UNIVERSE_SIZE   0x64400009
#define MPI_LASTUSEDCODE    0x6440000b
#define MPI_APPNUM          0x6440000d

int
MPIAPI
MPI_Comm_get_attr(
    MPI_Comm comm,
    int comm_keyval,
    _Out_ void* attribute_val,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Comm_get_attr(
    MPI_Comm comm,
    int comm_keyval,
    _Out_ void* attribute_val,
    _Out_ int* flag
    );

int
MPIAPI
MPI_Comm_delete_attr(
    MPI_Comm comm,
    int comm_keyval
    );
int
MPIAPI
PMPI_Comm_delete_attr(
    MPI_Comm comm,
    int comm_keyval
    );


typedef
int
(MPIAPI MPI_Win_copy_attr_function)(
    MPI_Win oldwin,
    int win_keyval,
    _In_opt_ void* extra_state,
    _In_ void* attribute_val_in,
    _Out_ void* attribute_val_out,
    _Out_ int* flag
    );

typedef
int
(MPIAPI MPI_Win_delete_attr_function)(
    MPI_Win win,
    int win_keyval,
    _In_ void* attribute_val,
    _In_opt_ void* extra_state
    );

#define MPI_WIN_NULL_COPY_FN ((MPI_Win_copy_attr_function*)0)
#define MPI_WIN_NULL_DELETE_FN ((MPI_Win_delete_attr_function*)0)
#define MPI_WIN_DUP_FN ((MPI_Win_copy_attr_function*)MPIR_Dup_fn)

int
MPIAPI
MPI_Win_create_keyval(
    _In_ MPI_Win_copy_attr_function* win_copy_attr_fn,
    _In_ MPI_Win_delete_attr_function* win_delete_attr_fn,
    _Out_ int* win_keyval,
    _In_opt_ void* extra_state
    );
int
MPIAPI
PMPI_Win_create_keyval(
    _In_ MPI_Win_copy_attr_function* win_copy_attr_fn,
    _In_ MPI_Win_delete_attr_function* win_delete_attr_fn,
    _Out_ int* win_keyval,
    _In_opt_ void* extra_state
    );

int
MPIAPI
MPI_Win_free_keyval(
    _Inout_ int* win_keyval
    );
int
MPIAPI
PMPI_Win_free_keyval(
    _Inout_ int* win_keyval
    );

int
MPIAPI
MPI_Win_set_attr(
    MPI_Win win,
    int win_keyval,
    _In_ void* attribute_val
    );
int
MPIAPI
PMPI_Win_set_attr(
    MPI_Win win,
    int win_keyval,
    _In_ void* attribute_val
    );


/* Predefined window key value attributes */
#define MPI_WIN_BASE        0x66000001
#define MPI_WIN_SIZE        0x66000003
#define MPI_WIN_DISP_UNIT   0x66000005

int
MPIAPI
MPI_Win_get_attr(
    MPI_Win win,
    int win_keyval,
    _Out_ void* attribute_val,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Win_get_attr(
    MPI_Win win,
    int win_keyval,
    _Out_ void* attribute_val,
    _Out_ int* flag
    );

int
MPIAPI
MPI_Win_delete_attr(
    MPI_Win win,
    int win_keyval
    );
int
MPIAPI
PMPI_Win_delete_attr(
    MPI_Win win,
    int win_keyval
    );


typedef
int
(MPIAPI MPI_Type_copy_attr_function)(
    MPI_Datatype olddatatype,
    int datatype_keyval,
    _In_opt_ void* extra_state,
    _In_ void* attribute_val_in,
    _Out_ void* attribute_val_out,
    _Out_ int* flag
    );

typedef
int
(MPIAPI MPI_Type_delete_attr_function)(
    MPI_Datatype datatype,
    int datatype_keyval,
    _In_ void* attribute_val,
    _In_opt_ void* extra_state
    );

#define MPI_TYPE_NULL_COPY_FN ((MPI_Type_copy_attr_function*)0)
#define MPI_TYPE_NULL_DELETE_FN ((MPI_Type_delete_attr_function*)0)
#define MPI_TYPE_DUP_FN ((MPI_Type_copy_attr_function*)MPIR_Dup_fn)

int
MPIAPI
MPI_Type_create_keyval(
    _In_ MPI_Type_copy_attr_function* type_copy_attr_fn,
    _In_ MPI_Type_delete_attr_function* type_delete_attr_fn,
    _Out_ int* type_keyval,
    _In_opt_ void* extra_state
    );
int
MPIAPI
PMPI_Type_create_keyval(
    _In_ MPI_Type_copy_attr_function* type_copy_attr_fn,
    _In_ MPI_Type_delete_attr_function* type_delete_attr_fn,
    _Out_ int* type_keyval,
    _In_opt_ void* extra_state
    );

int
MPIAPI
MPI_Type_free_keyval(
    _Inout_ int* type_keyval
    );
int
MPIAPI
PMPI_Type_free_keyval(
    _Inout_ int* type_keyval
    );

int
MPIAPI
MPI_Type_set_attr(
    MPI_Datatype type,
    int type_keyval,
    _In_ void* attribute_val
    );
int
MPIAPI
PMPI_Type_set_attr(
    MPI_Datatype type,
    int type_keyval,
    _In_ void* attribute_val
    );

int
MPIAPI
MPI_Type_get_attr(
    MPI_Datatype type,
    int type_keyval,
    _Out_ void* attribute_val,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Type_get_attr(
    MPI_Datatype type,
    int type_keyval,
    _Out_ void* attribute_val,
    _Out_ int* flag
    );

int
MPIAPI
MPI_Type_delete_attr(
    MPI_Datatype datatype,
    int type_keyval
    );
int
MPIAPI
PMPI_Type_delete_attr(
    MPI_Datatype datatype,
    int type_keyval
    );


/*---------------------------------------------*/
/* Section 6.8: Naming Objects                 */
/*---------------------------------------------*/

#define MPI_MAX_OBJECT_NAME 128

int
MPIAPI
MPI_Comm_set_name(
    MPI_Comm comm,
    _In_z_ char* comm_name
    );
int
MPIAPI
PMPI_Comm_set_name(
    MPI_Comm comm,
    _In_z_ char* comm_name
    );

int
MPIAPI
MPI_Comm_get_name(
    MPI_Comm comm,
    _Out_z_cap_post_count_(MPI_MAX_OBJECT_NAME,*resultlen) char* comm_name,
    _Out_ int* resultlen
    );
int
MPIAPI
PMPI_Comm_get_name(
    MPI_Comm comm,
    _Out_z_cap_post_count_(MPI_MAX_OBJECT_NAME,*resultlen) char* comm_name,
    _Out_ int* resultlen
    );

int
MPIAPI
MPI_Type_set_name(
    MPI_Datatype type,
    _In_z_ char* type_name
    );
int
MPIAPI
PMPI_Type_set_name(
    MPI_Datatype type,
    _In_z_ char* type_name
    );

int
MPIAPI
MPI_Type_get_name(
    MPI_Datatype type,
    _Out_z_cap_post_count_(MPI_MAX_OBJECT_NAME,*resultlen) char* type_name,
    _Out_ int* resultlen
    );
int
MPIAPI
PMPI_Type_get_name(
    MPI_Datatype type,
    _Out_z_cap_post_count_(MPI_MAX_OBJECT_NAME,*resultlen) char* type_name,
    _Out_ int* resultlen
    );

int
MPIAPI
MPI_Win_set_name(
    MPI_Win win,
    _In_z_ char* win_name
    );
int
MPIAPI
PMPI_Win_set_name(
    MPI_Win win,
    _In_z_ char* win_name
    );

int
MPIAPI
MPI_Win_get_name(
    MPI_Win win,
    _Out_z_cap_post_count_(MPI_MAX_OBJECT_NAME,*resultlen) char* win_name,
    _Out_ int* resultlen
    );
int
MPIAPI
PMPI_Win_get_name(
    MPI_Win win,
    _Out_z_cap_post_count_(MPI_MAX_OBJECT_NAME,*resultlen) char* win_name,
    _Out_ int* resultlen
    );


/*---------------------------------------------------------------------------*/
/* Chapter 7: Process Topologies                                             */
/*---------------------------------------------------------------------------*/

int
MPIAPI
MPI_Cart_create(
    MPI_Comm comm_old,
    int ndims,
    _In_count_(ndims) int* dims,
    _In_count_(ndims) int* periods,
    int reorder,
    _Out_ MPI_Comm* comm_cart
    );
int
MPIAPI
PMPI_Cart_create(
    MPI_Comm comm_old,
    int ndims,
    _In_count_(ndims) int* dims,
    _In_count_(ndims) int* periods,
    int reorder,
    _Out_ MPI_Comm* comm_cart
    );

int
MPIAPI
MPI_Dims_create(
    int nnodes,
    int ndims,
    _Inout_count_(ndims) int* dims
    );
int
MPIAPI
PMPI_Dims_create(
    int nnodes,
    int ndims,
    _Inout_count_(ndims) int* dims
    );

int
MPIAPI
MPI_Graph_create(
    MPI_Comm comm_old,
    int nnodes,
    _In_count_(nnodes) int* index,
    _In_ int* edges,
    int reorder,
    _Out_ MPI_Comm* comm_cart
    );
int
MPIAPI
PMPI_Graph_create(
    MPI_Comm comm_old,
    int nnodes,
    _In_count_(nnodes) int* index,
    _In_ int* edges,
    int reorder,
    _Out_ MPI_Comm* comm_cart
    );


/* Topology types */
enum
{
    MPI_GRAPH   = 1,
    MPI_CART    = 2
};

int
MPIAPI
MPI_Topo_test(
    MPI_Comm comm,
    _Out_ int* status
    );
int
MPIAPI
PMPI_Topo_test(
    MPI_Comm comm,
    _Out_ int* status
    );

int
MPIAPI
MPI_Graphdims_get(
    MPI_Comm comm,
    _Out_ int* nnodes,
    _Out_ int* nedges
    );
int
MPIAPI
PMPI_Graphdims_get(
    MPI_Comm comm,
    _Out_ int* nnodes,
    _Out_ int* nedges
    );

int
MPIAPI
MPI_Graph_get(
    MPI_Comm comm,
    int maxindex,
    int maxedges,
    _Out_cap_(maxindex) int* index,
    _Out_cap_(maxedges) int* edges
    );
int
MPIAPI
PMPI_Graph_get(
    MPI_Comm comm,
    int maxindex,
    int maxedges,
    _Out_cap_(maxindex) int* index,
    _Out_cap_(maxedges) int* edges
    );

int
MPIAPI
MPI_Cartdim_get(
    MPI_Comm comm,
    _Out_ int* ndims
    );
int
MPIAPI
PMPI_Cartdim_get(
    MPI_Comm comm,
    _Out_ int* ndims
    );

int
MPIAPI
MPI_Cart_get(
    MPI_Comm comm,
    int maxdims,
    _Out_cap_(maxdims) int* dims,
    _Out_cap_(maxdims) int* periods,
    _Out_cap_(maxdims) int* coords
    );
int
MPIAPI
PMPI_Cart_get(
    MPI_Comm comm,
    int maxdims,
    _Out_cap_(maxdims) int* dims,
    _Out_cap_(maxdims) int* periods,
    _Out_cap_(maxdims) int* coords
    );

int
MPIAPI
MPI_Cart_rank(
    MPI_Comm comm,
    _In_ int* coords,
    _Out_ int* rank
    );
int
MPIAPI
PMPI_Cart_rank(
    MPI_Comm comm,
    _In_ int* coords,
    _Out_ int* rank
    );

int
MPIAPI
MPI_Cart_coords(
    MPI_Comm comm,
    int rank,
    int maxdims,
    _Out_cap_(maxdims) int* coords
    );
int
MPIAPI
PMPI_Cart_coords(
    MPI_Comm comm,
    int rank,
    int maxdims,
    _Out_cap_(maxdims) int* coords
    );

int
MPIAPI
MPI_Graph_neighbors_count(
    MPI_Comm comm,
    int rank,
    _Out_ int* nneighbors
    );
int
MPIAPI
PMPI_Graph_neighbors_count(
    MPI_Comm comm,
    int rank,
    _Out_ int* nneighbors
    );

int
MPIAPI
MPI_Graph_neighbors(
    MPI_Comm comm,
    int rank,
    int maxneighbors,
    _Out_cap_(maxneighbors) int* neighbors
    );
int
MPIAPI
PMPI_Graph_neighbors(
    MPI_Comm comm,
    int rank,
    int maxneighbors,
    _Out_cap_(maxneighbors) int* neighbors
    );

int
MPIAPI
MPI_Cart_shift(
    MPI_Comm comm,
    int direction,
    int disp,
    _Out_ int* rank_source,
    _Out_ int* rank_dest
    );
int
MPIAPI
PMPI_Cart_shift(
    MPI_Comm comm,
    int direction,
    int disp,
    _Out_ int* rank_source,
    _Out_ int* rank_dest
    );

int
MPIAPI
MPI_Cart_sub(
    MPI_Comm comm,
    _In_ int* remain_dims,
    _Out_ MPI_Comm* newcomm
    );
int
MPIAPI
PMPI_Cart_sub(
    MPI_Comm comm,
    _In_ int* remain_dims,
    _Out_ MPI_Comm* newcomm
    );

int
MPIAPI
MPI_Cart_map(
    MPI_Comm comm,
    int ndims,
    _In_count_(ndims) int* dims,
    _In_count_(ndims) int* periods,
    _Out_ int* newrank
    );
int
MPIAPI
PMPI_Cart_map(
    MPI_Comm comm,
    int ndims,
    _In_count_(ndims) int* dims,
    _In_count_(ndims) int* periods,
    _Out_ int* newrank
    );

int
MPIAPI
MPI_Graph_map(
    MPI_Comm comm,
    int nnodes,
    _In_count_(nnodes) int* index,
    _In_ int* edges,
    _Out_ int* newrank
    );
int
MPIAPI
PMPI_Graph_map(
    MPI_Comm comm,
    int nnodes,
    _In_count_(nnodes) int* index,
    _In_ int* edges,
    _Out_ int* newrank
    );


/*---------------------------------------------------------------------------*/
/* Chapter 8: Environmental Management                                       */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------*/
/* Section 8.1: Implementation Information     */
/*---------------------------------------------*/

#define MPI_VERSION     2
#define MPI_SUBVERSION  0

int
MPIAPI
MPI_Get_version(
    _Out_ int* version,
    _Out_ int* subversion
    );
int
MPIAPI
PMPI_Get_version(
    _Out_ int* version,
    _Out_ int* subversion
    );

#define MPI_MAX_PROCESSOR_NAME  128

int
MPIAPI
MPI_Get_processor_name(
    _Out_z_cap_post_count_(MPI_MAX_PROCESSOR_NAME,*resultlen) char* name,
    _Out_ int* resultlen
    );
int
MPIAPI
PMPI_Get_processor_name(
    _Out_z_cap_post_count_(MPI_MAX_PROCESSOR_NAME,*resultlen) char* name,
    _Out_ int* resultlen
    );

/*---------------------------------------------*/
/* Section 8.2: Memory Allocation              */
/*---------------------------------------------*/

int
MPIAPI
MPI_Alloc_mem(
    MPI_Aint size,
    MPI_Info info,
    _Out_ void* baseptr
    );
int
MPIAPI
PMPI_Alloc_mem(
    MPI_Aint size,
    MPI_Info info,
    _Out_ void* baseptr
    );

int
MPIAPI
MPI_Free_mem(
    _In_ void* base
    );
int
MPIAPI
PMPI_Free_mem(
    _In_ void* base
    );


/*---------------------------------------------*/
/* Section 8.3: Error Handling                 */
/*---------------------------------------------*/

typedef
void
(MPIAPI MPI_Comm_errhandler_fn)(
    _In_ MPI_Comm* comm,
    _Inout_ int* errcode,
    ...
    );

int
MPIAPI
MPI_Comm_create_errhandler(
    _In_ MPI_Comm_errhandler_fn* function,
    _Out_ MPI_Errhandler* errhandler
    );
int
MPIAPI
PMPI_Comm_create_errhandler(
    _In_ MPI_Comm_errhandler_fn* function,
    _Out_ MPI_Errhandler* errhandler
    );

int
MPIAPI
MPI_Comm_set_errhandler(
    MPI_Comm comm,
    MPI_Errhandler errhandler
    );
int
MPIAPI
PMPI_Comm_set_errhandler(
    MPI_Comm comm,
    MPI_Errhandler errhandler
    );

int
MPIAPI
MPI_Comm_get_errhandler(
    MPI_Comm comm,
    _Out_ MPI_Errhandler* errhandler
    );
int
MPIAPI
PMPI_Comm_get_errhandler(
    MPI_Comm comm,
    _Out_ MPI_Errhandler* errhandler
    );


typedef
void
(MPIAPI MPI_Win_errhandler_fn)(
    _In_ MPI_Win* win,
    _Inout_ int* errcode,
    ...
    );

int
MPIAPI
MPI_Win_create_errhandler(
    _In_ MPI_Win_errhandler_fn* function,
    _Out_ MPI_Errhandler* errhandler
    );
int
MPIAPI
PMPI_Win_create_errhandler(
    _In_ MPI_Win_errhandler_fn* function,
    _Out_ MPI_Errhandler* errhandler
    );

int
MPIAPI
MPI_Win_set_errhandler(
    MPI_Win win,
    MPI_Errhandler errhandler
    );
int
MPIAPI
PMPI_Win_set_errhandler(
    MPI_Win win,
    MPI_Errhandler errhandler
    );

int
MPIAPI
MPI_Win_get_errhandler(
    MPI_Win win,
    _Out_ MPI_Errhandler* errhandler
    );
int
MPIAPI
PMPI_Win_get_errhandler(
    MPI_Win win,
    _Out_ MPI_Errhandler* errhandler
    );


typedef
void
(MPIAPI MPI_File_errhandler_fn)(
    _In_ MPI_File* file,
    _Inout_ int* errcode,
    ...
    );

int
MPIAPI
MPI_File_create_errhandler(
    _In_ MPI_File_errhandler_fn* function,
    _Out_ MPI_Errhandler* errhandler
    );
int
MPIAPI
PMPI_File_create_errhandler(
    _In_ MPI_File_errhandler_fn* function,
    _Out_ MPI_Errhandler* errhandler
    );

int
MPIAPI
MPI_File_set_errhandler(
    MPI_File file,
    MPI_Errhandler errhandler
    );
int
MPIAPI
PMPI_File_set_errhandler(
    MPI_File file,
    MPI_Errhandler errhandler
    );

int
MPIAPI
MPI_File_get_errhandler(
    MPI_File file,
    _Out_ MPI_Errhandler* errhandler
    );
int
MPIAPI
PMPI_File_get_errhandler(
    MPI_File file,
    _Out_ MPI_Errhandler* errhandler
    );

int
MPIAPI
MPI_Errhandler_free(
    _Inout_ MPI_Errhandler* errhandler
    );
int
MPIAPI
PMPI_Errhandler_free(
    _Inout_ MPI_Errhandler* errhandler
    );

#define MPI_MAX_ERROR_STRING    512

int
MPIAPI
MPI_Error_string(
    int errorcode,
    _Out_z_cap_post_count_(MPI_MAX_ERROR_STRING,*resultlen) char* string,
    _Out_ int* resultlen
    );
int
MPIAPI
PMPI_Error_string(
    int errorcode,
    _Out_z_cap_post_count_(MPI_MAX_ERROR_STRING,*resultlen) char* string,
    _Out_ int* resultlen
    );


/*---------------------------------------------*/
/* Section 8.4: Error Codes and Classes        */
/*---------------------------------------------*/

int
MPIAPI
MPI_Error_class(
    int errorcode,
    _Out_ int* errorclass
    );
int
MPIAPI
PMPI_Error_class(
    int errorcode,
    _Out_ int* errorclass
    );

int
MPIAPI
MPI_Add_error_class(
    _Out_ int* errorclass
    );
int
MPIAPI
PMPI_Add_error_class(
    _Out_ int* errorclass
    );

int
MPIAPI
MPI_Add_error_code(
    int errorclass,
    _Out_ int* errorcode
    );
int
MPIAPI
PMPI_Add_error_code(
    int errorclass,
    _Out_ int* errorcode
    );

int
MPIAPI
MPI_Add_error_string(
    int errorcode,
    _In_z_ char* string
    );
int
MPIAPI
PMPI_Add_error_string(
    int errorcode,
    _In_z_ char* string
    );

int
MPIAPI
MPI_Comm_call_errhandler(
    MPI_Comm comm,
    int errorcode
    );
int
MPIAPI
PMPI_Comm_call_errhandler(
    MPI_Comm comm,
    int errorcode
    );

int
MPIAPI
MPI_Win_call_errhandler(
    MPI_Win win,
    int errcode
    );
int
MPIAPI
PMPI_Win_call_errhandler(
    MPI_Win win,
    int errcode
    );

int
MPIAPI
MPI_File_call_errhandler(
    MPI_File file,
    int errorcode
    );
int
MPIAPI
PMPI_File_call_errhandler(
    MPI_File file,
    int errorcode
    );


/*---------------------------------------------*/
/* Section 8.6: Timers and Synchronization     */
/*---------------------------------------------*/

double
MPIAPI
MPI_Wtime(
    void
    );
double
MPIAPI
PMPI_Wtime(
    void
    );

double
MPIAPI
MPI_Wtick(
    void
    );
double
MPIAPI
PMPI_Wtick(
    void
    );


/*---------------------------------------------*/
/* Section 8.7: Startup                        */
/*---------------------------------------------*/

int
MPIAPI
MPI_Init(
    _In_opt_ int* argc,
    _In_opt_count_(*argc) char*** argv
    );
int
MPIAPI
PMPI_Init(
    _In_opt_ int* argc,
    _In_opt_count_(*argc) char*** argv
    );

int
MPIAPI
MPI_Finalize(
    void
    );
int
MPIAPI
PMPI_Finalize(
    void
    );

int
MPIAPI
MPI_Initialized(
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Initialized(
    _Out_ int* flag
    );

int
MPIAPI
MPI_Abort(
    MPI_Comm comm,
    int errorcode
    );
int
MPIAPI
PMPI_Abort(
    MPI_Comm comm,
    int errorcode
    );

int
MPIAPI
MPI_Finalized(
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Finalized(
    _Out_ int* flag
    );


/*---------------------------------------------------------------------------*/
/* Chapter 9: The Info Object                                                */
/*---------------------------------------------------------------------------*/

#define MPI_MAX_INFO_KEY    255
#define MPI_MAX_INFO_VAL   1024

int
MPIAPI
MPI_Info_create(
    _Out_ MPI_Info* info
    );
int
MPIAPI
PMPI_Info_create(
    _Out_ MPI_Info* info
    );

int
MPIAPI
MPI_Info_set(
    MPI_Info info,
    _In_z_ char* key,
    _In_z_ char* value
    );
int
MPIAPI
PMPI_Info_set(
    MPI_Info info,
    _In_z_ char* key,
    _In_z_ char* value
    );

int
MPIAPI
MPI_Info_delete(
    MPI_Info info,
    _In_z_ char* key
    );
int
MPIAPI
PMPI_Info_delete(
    MPI_Info info,
    _In_z_ char* key
    );

int
MPIAPI
MPI_Info_get(
    MPI_Info info,
    _In_z_ char* key,
    int valuelen,
    _Out_z_cap_(valuelen) char* value,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Info_get(
    MPI_Info info,
    _In_z_ char* key,
    int valuelen,
    _Out_z_cap_(valuelen) char* value,
    _Out_ int* flag
    );

int
MPIAPI
MPI_Info_get_valuelen(
    MPI_Info info,
    _In_z_ char* key,
    _Out_ int* valuelen,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Info_get_valuelen(
    MPI_Info info,
    _In_z_ char* key,
    _Out_ int* valuelen,
    _Out_ int* flag
    );

int
MPIAPI
MPI_Info_get_nkeys(
    MPI_Info info,
    _Out_ int* nkeys
    );
int
MPIAPI
PMPI_Info_get_nkeys(
    MPI_Info info,
    _Out_ int* nkeys
    );

int
MPIAPI
MPI_Info_get_nthkey(
    MPI_Info info,
    int n,
    _Out_z_cap_(MPI_MAX_INFO_KEY) char* key
    );
int
MPIAPI
PMPI_Info_get_nthkey(
    MPI_Info info,
    int n,
    _Out_z_cap_(MPI_MAX_INFO_KEY) char* key
    );

int
MPIAPI
MPI_Info_dup(
    MPI_Info info,
    _Out_ MPI_Info* newinfo
    );
int
MPIAPI
PMPI_Info_dup(
    MPI_Info info,
    _Out_ MPI_Info* newinfo
    );

int
MPIAPI
MPI_Info_free(
    _Inout_ MPI_Info* info
    );
int
MPIAPI
PMPI_Info_free(
    _Inout_ MPI_Info* info
    );


/*---------------------------------------------------------------------------*/
/* Chapter 10: Process Creation and Management                               */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------*/
/* Section 10.3: Process Manager Interface     */
/*---------------------------------------------*/

#define MPI_ARGV_NULL ((char**)0)
#define MPI_ARGVS_NULL ((char***)0)

#define MPI_ERRCODES_IGNORE ((int*)0)

int
MPIAPI
MPI_Comm_spawn(
    _In_z_ char* command,
    _In_ char* argv[],
    int maxprocs,
    MPI_Info info,
    int root,
    MPI_Comm comm,
    _Out_ MPI_Comm* intercomm,
    _Out_opt_cap_(maxprocs) int array_of_errcodes[]
    );
int
MPIAPI
PMPI_Comm_spawn(
    _In_z_ char* command,
    _In_ char* argv[],
    int maxprocs,
    MPI_Info info,
    int root,
    MPI_Comm comm,
    _Out_ MPI_Comm* intercomm,
    _Out_opt_cap_(maxprocs) int array_of_errcodes[]
    );

int
MPIAPI
MPI_Comm_get_parent(
    _Out_ MPI_Comm* parent
    );
int
MPIAPI
PMPI_Comm_get_parent(
    _Out_ MPI_Comm* parent
    );

int
MPIAPI
MPI_Comm_spawn_multiple(
    int count,
    _In_count_(count) char* array_of_commands[],
    _In_opt_count_(count) char** array_of_argv[],
    _In_count_(count) int array_of_maxprocs[],
    _In_count_(count) MPI_Info array_of_info[],
    int root,
    MPI_Comm comm,
    _Out_ MPI_Comm* intercomm,
    _Out_opt_ int array_of_errcodes[]
    );
int
MPIAPI
PMPI_Comm_spawn_multiple(
    int count,
    _In_count_(count) char* array_of_commands[],
    _In_opt_count_(count) char** array_of_argv[],
    _In_count_(count) int array_of_maxprocs[],
    _In_count_(count) MPI_Info array_of_info[],
    int root,
    MPI_Comm comm,
    _Out_ MPI_Comm* intercomm,
    _Out_opt_ int array_of_errcodes[]
    );


/*---------------------------------------------*/
/* Section 10.4: Establishing Communication    */
/*---------------------------------------------*/

#define MPI_MAX_PORT_NAME   256

int
MPIAPI
MPI_Open_port(
    MPI_Info info,
    _Out_cap_(MPI_MAX_PORT_NAME) char* port_name
    );
int
MPIAPI
PMPI_Open_port(
    MPI_Info info,
    _Out_cap_(MPI_MAX_PORT_NAME) char* port_name
    );

int
MPIAPI
MPI_Close_port(
    _In_z_ char* port_name
    );
int
MPIAPI
PMPI_Close_port(
    _In_z_ char* port_name
    );

int
MPIAPI
MPI_Comm_accept(
    _In_z_ char* port_name,
    MPI_Info info,
    int root,
    MPI_Comm comm,
    _Out_ MPI_Comm* newcomm
    );
int
MPIAPI
PMPI_Comm_accept(
    _In_z_ char* port_name,
    MPI_Info info,
    int root,
    MPI_Comm comm,
    _Out_ MPI_Comm* newcomm
    );

int
MPIAPI
MPI_Comm_connect(
    _In_z_ char* port_name,
    MPI_Info info,
    int root,
    MPI_Comm comm,
    _Out_ MPI_Comm* newcomm
    );
int
MPIAPI
PMPI_Comm_connect(
    _In_z_ char* port_name,
    MPI_Info info,
    int root,
    MPI_Comm comm,
    _Out_ MPI_Comm* newcomm
    );


/*---------------------------------------------*/
/* Section 10.4.4: Name Publishing             */
/*---------------------------------------------*/

int
MPIAPI
MPI_Publish_name(
    _In_z_ char* service_name,
    MPI_Info info,
    _In_z_ char* port_name
    );
int
MPIAPI
PMPI_Publish_name(
    _In_z_ char* service_name,
    MPI_Info info,
    _In_z_ char* port_name
    );

int
MPIAPI
MPI_Unpublish_name(
    _In_z_ char* service_name,
    MPI_Info info,
    _In_z_ char* port_name
    );
int
MPIAPI
PMPI_Unpublish_name(
    _In_z_ char* service_name,
    MPI_Info info,
    _In_z_ char* port_name
    );

int
MPIAPI
MPI_Lookup_name(
    _In_z_ char* service_name,
    MPI_Info info,
    _Out_cap_(MPI_MAX_PORT_NAME) char* port_name
    );
int
MPIAPI
PMPI_Lookup_name(
    _In_z_ char* service_name,
    MPI_Info info,
    _Out_cap_(MPI_MAX_PORT_NAME) char* port_name
    );


/*---------------------------------------------*/
/* Section 10.5: Other Functionality           */
/*---------------------------------------------*/

int
MPIAPI
MPI_Comm_disconnect(
    _In_ MPI_Comm* comm
    );
int
MPIAPI
PMPI_Comm_disconnect(
    _In_ MPI_Comm* comm
    );

int
MPIAPI
MPI_Comm_join(
    int fd,
    _Out_ MPI_Comm* intercomm
    );
int
MPIAPI
PMPI_Comm_join(
    int fd,
    _Out_ MPI_Comm* intercomm
    );


/*---------------------------------------------------------------------------*/
/* Chapter 11: One-Sided Communications                                      */
/*---------------------------------------------------------------------------*/

int
MPIAPI
MPI_Win_create(
    _In_ void* base,
    MPI_Aint size,
    int disp_unit,
    MPI_Info info,
    MPI_Comm comm,
    _Out_ MPI_Win* win
    );
int
MPIAPI
PMPI_Win_create(
    _In_ void* base,
    MPI_Aint size,
    int disp_unit,
    MPI_Info info,
    MPI_Comm comm,
    _Out_ MPI_Win* win
    );

int
MPIAPI
MPI_Win_free(
    _Inout_ MPI_Win* win
    );
int
MPIAPI
PMPI_Win_free(
    _Inout_ MPI_Win* win
    );

int
MPIAPI
MPI_Win_get_group(
    MPI_Win win,
    _Out_ MPI_Group* group
    );
int
MPIAPI
PMPI_Win_get_group(
    MPI_Win win,
    _Out_ MPI_Group* group
    );

int
MPIAPI
MPI_Put(
    _In_ void* origin_addr,
    int origin_count,
    MPI_Datatype origin_datatype,
    int target_rank,
    MPI_Aint target_disp,
    int target_count,
    MPI_Datatype datatype,
    MPI_Win win
    );
int
MPIAPI
PMPI_Put(
    _In_ void* origin_addr,
    int origin_count,
    MPI_Datatype origin_datatype,
    int target_rank,
    MPI_Aint target_disp,
    int target_count,
    MPI_Datatype datatype,
    MPI_Win win
    );

int
MPIAPI
MPI_Get(
    _Out_ void* origin_addr,
    int origin_count,
    MPI_Datatype origin_datatype,
    int target_rank,
    MPI_Aint target_disp,
    int target_count,
    MPI_Datatype datatype,
    MPI_Win win
    );
int
MPIAPI
PMPI_Get(
    _Out_ void* origin_addr,
    int origin_count,
    MPI_Datatype origin_datatype,
    int target_rank,
    MPI_Aint target_disp,
    int target_count,
    MPI_Datatype datatype,
    MPI_Win win
    );

int
MPIAPI
MPI_Accumulate(
    _In_ void* origin_addr,
    int origin_count,
    MPI_Datatype origin_datatype,
    int target_rank,
    MPI_Aint target_disp,
    int target_count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Win win
    );
int
MPIAPI
PMPI_Accumulate(
    _In_ void* origin_addr,
    int origin_count,
    MPI_Datatype origin_datatype,
    int target_rank,
    MPI_Aint target_disp,
    int target_count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Win win
    );

/* Asserts for one-sided communication */
#define MPI_MODE_NOCHECK    1024
#define MPI_MODE_NOSTORE    2048
#define MPI_MODE_NOPUT      4096
#define MPI_MODE_NOPRECEDE  8192
#define MPI_MODE_NOSUCCEED 16384

int
MPIAPI
MPI_Win_fence(
    int assert,
    MPI_Win win
    );
int
MPIAPI
PMPI_Win_fence(
    int assert,
    MPI_Win win
    );

int
MPIAPI
MPI_Win_start(
    MPI_Group group,
    int assert,
    MPI_Win win
    );
int
MPIAPI
PMPI_Win_start(
    MPI_Group group,
    int assert,
    MPI_Win win
    );

int
MPIAPI
MPI_Win_complete(
    MPI_Win win
    );
int
MPIAPI
PMPI_Win_complete(
    MPI_Win win
    );

int
MPIAPI
MPI_Win_post(
    MPI_Group group,
    int assert,
    MPI_Win win
    );
int
MPIAPI
PMPI_Win_post(
    MPI_Group group,
    int assert,
    MPI_Win win
    );

int
MPIAPI
MPI_Win_wait(
    MPI_Win win
    );
int
MPIAPI
PMPI_Win_wait(
    MPI_Win win
    );

int
MPIAPI
MPI_Win_test(
    MPI_Win win,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Win_test(
    MPI_Win win,
    _Out_ int* flag
    );

#define MPI_LOCK_EXCLUSIVE  234
#define MPI_LOCK_SHARED     235

int
MPIAPI
MPI_Win_lock(
    int lock_type,
    int rank,
    int assert,
    MPI_Win win
    );
int
MPIAPI
PMPI_Win_lock(
    int lock_type,
    int rank,
    int assert,
    MPI_Win win
    );

int
MPIAPI
MPI_Win_unlock(
    int rank,
    MPI_Win win
    );
int
MPIAPI
PMPI_Win_unlock(
    int rank,
    MPI_Win win
    );


/*---------------------------------------------------------------------------*/
/* Chapter 12: External Interfaces                                           */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------*/
/* Section 12.2: Generalized Requests          */
/*---------------------------------------------*/

typedef
int
(MPIAPI MPI_Grequest_query_function)(
    _In_opt_ void* extra_state,
    _Out_ MPI_Status* status
    );

typedef
int
(MPIAPI MPI_Grequest_free_function)(
    _In_opt_ void* extra_state
    );

typedef
int
(MPIAPI MPI_Grequest_cancel_function)(
    _In_opt_ void* extra_state,
    int complete
    );

int
MPIAPI
MPI_Grequest_start(
    _In_ MPI_Grequest_query_function* query_fn,
    _In_ MPI_Grequest_free_function* free_fn,
    _In_ MPI_Grequest_cancel_function* cancel_fn,
    _In_opt_ void* extra_state,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_Grequest_start(
    _In_ MPI_Grequest_query_function* query_fn,
    _In_ MPI_Grequest_free_function* free_fn,
    _In_ MPI_Grequest_cancel_function* cancel_fn,
    _In_opt_ void* extra_state,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_Grequest_complete(
    MPI_Request request
    );
int
MPIAPI
PMPI_Grequest_complete(
    MPI_Request request
    );


/*---------------------------------------------*/
/* Section 12.3: Information with Status       */
/*---------------------------------------------*/

int
MPIAPI
MPI_Status_set_elements(
    _In_ MPI_Status* status,
    MPI_Datatype datatype,
    int count
    );
int
MPIAPI
PMPI_Status_set_elements(
    _In_ MPI_Status* status,
    MPI_Datatype datatype,
    int count
    );

int
MPIAPI
MPI_Status_set_cancelled(
    _In_ MPI_Status* status,
    int flag
    );
int
MPIAPI
PMPI_Status_set_cancelled(
    _In_ MPI_Status* status,
    int flag
    );


/*---------------------------------------------*/
/* Section 12.4: Threads                       */
/*---------------------------------------------*/

#define MPI_THREAD_SINGLE       0
#define MPI_THREAD_FUNNELED     1
#define MPI_THREAD_SERIALIZED   2
#define MPI_THREAD_MULTIPLE     3

int
MPIAPI
MPI_Init_thread(
    _In_opt_ int* argc,
    _In_opt_count_(*argc) char*** argv,
    int required,
    _Out_ int* provided
    );
int
MPIAPI
PMPI_Init_thread(
    _In_opt_ int* argc,
    _In_opt_count_(*argc) char*** argv,
    int required,
    _Out_ int* provided
    );

int
MPIAPI
MPI_Query_thread(
    _Out_ int* provided
    );
int
MPIAPI
PMPI_Query_thread(
    _Out_ int* provided
    );

int
MPIAPI
MPI_Is_thread_main(
    _Out_ int* flag
    );
int
MPIAPI
PMPI_Is_thread_main(
    _Out_ int* flag
    );


/*---------------------------------------------------------------------------*/
/* Chapter 13: I/O                                                           */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------*/
/* Section 13.2: File Manipulation             */
/*---------------------------------------------*/

#define MPI_MODE_CREATE             0x00000001
#define MPI_MODE_RDONLY             0x00000002
#define MPI_MODE_WRONLY             0x00000004
#define MPI_MODE_RDWR               0x00000008
#define MPI_MODE_DELETE_ON_CLOSE    0x00000010
#define MPI_MODE_UNIQUE_OPEN        0x00000020
#define MPI_MODE_EXCL               0x00000040
#define MPI_MODE_APPEND             0x00000080
#define MPI_MODE_SEQUENTIAL         0x00000100
#define MSMPI_MODE_HIDDEN           0x00000200

int
MPIAPI
MPI_File_open(
    MPI_Comm comm,
    _In_z_ char* filename,
    int amode,
    MPI_Info info,
    _Out_ MPI_File* newfile
    );
int
MPIAPI
PMPI_File_open(
    MPI_Comm comm,
    _In_z_ char* filename,
    int amode,
    MPI_Info info,
    _Out_ MPI_File* newfile
    );

int
MPIAPI
MPI_File_close(
    _In_ MPI_File* file
    );
int
MPIAPI
PMPI_File_close(
    _In_ MPI_File* file
    );

int
MPIAPI
MPI_File_delete(
    _In_z_ char* filename,
    MPI_Info info
    );
int
MPIAPI
PMPI_File_delete(
    _In_z_ char* filename,
    MPI_Info info
    );

int
MPIAPI
MPI_File_set_size(
    MPI_File file,
    MPI_Offset size
    );
int
MPIAPI
PMPI_File_set_size(
    MPI_File file,
    MPI_Offset size
    );

int
MPIAPI
MPI_File_preallocate(
    MPI_File file,
    MPI_Offset size
    );
int
MPIAPI
PMPI_File_preallocate(
    MPI_File file,
    MPI_Offset size
    );

int
MPIAPI
MPI_File_get_size(
    MPI_File file,
    _Out_ MPI_Offset* size
    );
int
MPIAPI
PMPI_File_get_size(
    MPI_File file,
    _Out_ MPI_Offset* size
    );

int
MPIAPI
MPI_File_get_group(
    MPI_File file,
    _Out_ MPI_Group* group
    );
int
MPIAPI
PMPI_File_get_group(
    MPI_File file,
    _Out_ MPI_Group* group
    );

int
MPIAPI
MPI_File_get_amode(
    MPI_File file,
    _Out_ int* amode
    );
int
MPIAPI
PMPI_File_get_amode(
    MPI_File file,
    _Out_ int* amode
    );

int
MPIAPI
MPI_File_set_info(
    MPI_File file,
    MPI_Info info
    );
int
MPIAPI
PMPI_File_set_info(
    MPI_File file,
    MPI_Info info
    );

int
MPIAPI
MPI_File_get_info(
    MPI_File file,
    _Out_ MPI_Info* info
    );
int
MPIAPI
PMPI_File_get_info(
    MPI_File file,
    _Out_ MPI_Info* info
    );


/*---------------------------------------------*/
/* Section 13.3: File Views                    */
/*---------------------------------------------*/

#define MPI_DISPLACEMENT_CURRENT (-54278278)

int
MPIAPI
MPI_File_set_view(
    MPI_File file,
    MPI_Offset disp,
    MPI_Datatype etype,
    MPI_Datatype filetype,
    _In_z_ char* datarep,
    MPI_Info info
    );
int
MPIAPI
PMPI_File_set_view(
    MPI_File file,
    MPI_Offset disp,
    MPI_Datatype etype,
    MPI_Datatype filetype,
    _In_z_ char* datarep,
    MPI_Info info
    );

#define MPI_MAX_DATAREP_STRING  128

int
MPIAPI
MPI_File_get_view(
    MPI_File file,
    _Out_ MPI_Offset* disp,
    _Out_ MPI_Datatype* etype,
    _Out_ MPI_Datatype* filetype,
    _Out_z_cap_(MPI_MAX_DATAREP_STRING) char* datarep
    );
int
MPIAPI
PMPI_File_get_view(
    MPI_File file,
    _Out_ MPI_Offset* disp,
    _Out_ MPI_Datatype* etype,
    _Out_ MPI_Datatype* filetype,
    _Out_z_cap_(MPI_MAX_DATAREP_STRING) char* datarep
    );


/*---------------------------------------------*/
/* Section 13.4: Data Access                   */
/*---------------------------------------------*/

int
MPIAPI
MPI_File_read_at(
    MPI_File file,
    MPI_Offset offset,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_read_at(
    MPI_File file,
    MPI_Offset offset,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_read_at_all(
    MPI_File file,
    MPI_Offset offset,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_read_at_all(
    MPI_File file,
    MPI_Offset offset,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_write_at(
    MPI_File file,
    MPI_Offset offset,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write_at(
    MPI_File file,
    MPI_Offset offset,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_write_at_all(
    MPI_File file,
    MPI_Offset offset,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write_at_all(
    MPI_File file,
    MPI_Offset offset,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_iread_at(
    MPI_File file,
    MPI_Offset offset,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_File_iread_at(
    MPI_File file,
    MPI_Offset offset,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_File_iwrite_at(
    MPI_File file,
    MPI_Offset offset,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_File_iwrite_at(
    MPI_File file,
    MPI_Offset offset,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_File_read(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_read(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_read_all(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_read_all(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_write(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_write_all(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write_all(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );


int
MPIAPI
MPI_File_iread(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_File_iread(
    _In_ MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_File_iwrite(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_File_iwrite(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );


/* File seek whence */
#define MPI_SEEK_SET    600
#define MPI_SEEK_CUR    602
#define MPI_SEEK_END    604

int
MPIAPI
MPI_File_seek(
    MPI_File file,
    MPI_Offset offset,
    int whence
    );
int
MPIAPI
PMPI_File_seek(
    MPI_File file,
    MPI_Offset offset,
    int whence
    );

int
MPIAPI
MPI_File_get_position(
    MPI_File file,
    _Out_ MPI_Offset* offset
    );
int
MPIAPI
PMPI_File_get_position(
    MPI_File file,
    _Out_ MPI_Offset* offset
    );

int
MPIAPI
MPI_File_get_byte_offset(
    MPI_File file,
    MPI_Offset offset,
    _Out_ MPI_Offset* disp
    );
int
MPIAPI
PMPI_File_get_byte_offset(
    MPI_File file,
    MPI_Offset offset,
    _Out_ MPI_Offset* disp
    );

int
MPIAPI
MPI_File_read_shared(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
     );
int
MPIAPI
PMPI_File_read_shared(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
     );

int
MPIAPI
MPI_File_write_shared(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write_shared(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_iread_shared(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_File_iread_shared(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_File_iwrite_shared(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );
int
MPIAPI
PMPI_File_iwrite_shared(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Request* request
    );

int
MPIAPI
MPI_File_read_ordered(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_read_ordered(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_write_ordered(
    _In_ MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write_ordered(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_seek_shared(
    MPI_File file,
    MPI_Offset offset,
    int whence
    );
int
MPIAPI
PMPI_File_seek_shared(
    MPI_File file,
    MPI_Offset offset,
    int whence
    );

int
MPIAPI
MPI_File_get_position_shared(
    MPI_File file,
    _Out_ MPI_Offset* offset
    );
int
MPIAPI
PMPI_File_get_position_shared(
    MPI_File file,
    _Out_ MPI_Offset* offset
    );

int
MPIAPI
MPI_File_read_at_all_begin(
    MPI_File file,
    MPI_Offset offset,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype
    );
int
MPIAPI
PMPI_File_read_at_all_begin(
    MPI_File file,
    MPI_Offset offset,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype
    );

int
MPIAPI
MPI_File_read_at_all_end(
    MPI_File file,
    _Out_ void* buf,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_read_at_all_end(
    MPI_File file,
    _Out_ void* buf,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_write_at_all_begin(
    MPI_File file,
    MPI_Offset offset,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype
    );
int
MPIAPI
PMPI_File_write_at_all_begin(
    MPI_File file,
    MPI_Offset offset,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype
    );

int
MPIAPI
MPI_File_write_at_all_end(
    MPI_File file,
    _In_ void* buf,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write_at_all_end(
    MPI_File file,
    _In_ void* buf,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_read_all_begin(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype
    );
int
MPIAPI
PMPI_File_read_all_begin(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype
    );

int
MPIAPI
MPI_File_read_all_end(
    MPI_File file,
    _Out_ void* buf,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_read_all_end(
    MPI_File file,
    _Out_ void* buf,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_write_all_begin(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype
    );
int
MPIAPI
PMPI_File_write_all_begin(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype
    );

int
MPIAPI
MPI_File_write_all_end(
    MPI_File file,
    _In_ void* buf,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write_all_end(
    MPI_File file,
    _In_ void* buf,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_read_ordered_begin(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype
    );
int
MPIAPI
PMPI_File_read_ordered_begin(
    MPI_File file,
    _Out_ void* buf,
    int count,
    MPI_Datatype datatype
    );

int
MPIAPI
MPI_File_read_ordered_end(
    MPI_File file,
    _Out_ void* buf,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_read_ordered_end(
    MPI_File file,
    _Out_ void* buf,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_File_write_ordered_begin(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype
    );
int
MPIAPI
PMPI_File_write_ordered_begin(
    MPI_File file,
    _In_ void* buf,
    int count,
    MPI_Datatype datatype
    );

int
MPIAPI
MPI_File_write_ordered_end(
    MPI_File file,
    _In_ void* buf,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_File_write_ordered_end(
    MPI_File file,
    _In_ void* buf,
    _Out_ MPI_Status* status
    );


/*---------------------------------------------*/
/* Section 13.5: File Interoperability         */
/*---------------------------------------------*/

int
MPIAPI
MPI_File_get_type_extent(
    MPI_File file,
    MPI_Datatype datatype,
    _Out_ MPI_Aint* extent
    );
int
MPIAPI
PMPI_File_get_type_extent(
    MPI_File file,
    MPI_Datatype datatype,
    _Out_ MPI_Aint* extent
    );


typedef
int
(MPIAPI MPI_Datarep_conversion_function)(
    _Inout_ void* userbuf,
    MPI_Datatype datatype,
    int count,
    _Inout_ void* filebuf,
    MPI_Offset position,
    _In_ void* extra_state
    );

typedef
int
(MPIAPI MPI_Datarep_extent_function)(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* file_extent,
    _In_ void* extra_state
    );

#define MPI_CONVERSION_FN_NULL ((MPI_Datarep_conversion_function*)0)

int
MPIAPI
MPI_Register_datarep(
    _In_z_ char* datarep,
    _In_opt_ MPI_Datarep_conversion_function* read_conversion_fn,
    _In_opt_ MPI_Datarep_conversion_function* write_conversion_fn,
    _In_ MPI_Datarep_extent_function* dtype_file_extent_fn,
    _In_opt_ void* extra_state
    );
int
MPIAPI
PMPI_Register_datarep(
    _In_z_ char* datarep,
    _In_opt_ MPI_Datarep_conversion_function* read_conversion_fn,
    _In_opt_ MPI_Datarep_conversion_function* write_conversion_fn,
    _In_ MPI_Datarep_extent_function* dtype_file_extent_fn,
    _In_opt_ void* extra_state
    );


/*---------------------------------------------*/
/* Section 13.6: Consistency and Semantics     */
/*---------------------------------------------*/

int
MPIAPI
MPI_File_set_atomicity(
    MPI_File file,
    int flag
    );
int
MPIAPI
PMPI_File_set_atomicity(
    MPI_File file,
    int flag
    );

int
MPIAPI
MPI_File_get_atomicity(
    MPI_File file,
    _Out_ int* flag
    );
int
MPIAPI
PMPI_File_get_atomicity(
    MPI_File file,
    _Out_ int* flag
    );

int
MPIAPI
MPI_File_sync(
    MPI_File file
    );
int
MPIAPI
PMPI_File_sync(
    MPI_File file
    );


/*---------------------------------------------------------------------------*/
/* Chapter 14: Profiling Interface                                           */
/*---------------------------------------------------------------------------*/

int
MPIAPI
MPI_Pcontrol(
    const int level,
    ...);
int
MPIAPI
PMPI_Pcontrol(
    const int level,
    ...);


/*---------------------------------------------------------------------------*/
/* Chapter 15: Depricated Functions                                          */
/*---------------------------------------------------------------------------*/

#ifdef MSMPI_NO_DEPRECATE_20
#define MSMPI_DEPRECATE_20( x )
#else
#define MSMPI_DEPRECATE_20( x ) __declspec(deprecated( \
    "Deprecated in MPI 2.0, use '" #x "'.  " \
    "To disable deprecation, define MSMPI_NO_DEPRECATE_20." ))
#endif

MSMPI_DEPRECATE_20( MPI_Type_create_hvector )
int
MPIAPI
MPI_Type_hvector(
    int count,
    int blocklength,
    MPI_Aint stride,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
MSMPI_DEPRECATE_20( PMPI_Type_create_hvector )
int
MPIAPI
PMPI_Type_hvector(
    int count,
    int blocklength,
    MPI_Aint stride,
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );

MSMPI_DEPRECATE_20( MPI_Type_create_hindexed )
int
MPIAPI
MPI_Type_hindexed(
    int count,
    _In_count_(count) int array_of_blocklengths[],
    _In_count_(count) MPI_Aint array_of_displacements[],
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );
MSMPI_DEPRECATE_20( PMPI_Type_create_hindexed )
int
MPIAPI
PMPI_Type_hindexed(
    int count,
    _In_count_(count) int array_of_blocklengths[],
    _In_count_(count) MPI_Aint array_of_displacements[],
    MPI_Datatype oldtype,
    _Out_ MPI_Datatype* newtype
    );

MSMPI_DEPRECATE_20( MPI_Type_create_struct )
int
MPIAPI
MPI_Type_struct(
    int count,
    _In_count_(count) int array_of_blocklengths[],
    _In_count_(count) MPI_Aint array_of_displacements[],
    _In_count_(count) MPI_Datatype array_of_types[],
    _In_ MPI_Datatype* newtype
    );
MSMPI_DEPRECATE_20( PMPI_Type_create_struct )
int
MPIAPI
PMPI_Type_struct(
    int count,
    _In_count_(count) int array_of_blocklengths[],
    _In_count_(count) MPI_Aint array_of_displacements[],
    _In_count_(count) MPI_Datatype array_of_types[],
    _In_ MPI_Datatype* newtype
    );

MSMPI_DEPRECATE_20( MPI_Get_address )
int
MPIAPI
MPI_Address(
    _In_ void* location,
    _Out_ MPI_Aint* address
    );
MSMPI_DEPRECATE_20( PMPI_Get_address )
int
MPIAPI
PMPI_Address(
    _In_ void* location,
    _Out_ MPI_Aint* address
    );

MSMPI_DEPRECATE_20( MPI_Type_get_extent )
int
MPIAPI
MPI_Type_extent(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* extent
    );
MSMPI_DEPRECATE_20( PMPI_Type_get_extent )
int
MPIAPI
PMPI_Type_extent(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* extent
    );

MSMPI_DEPRECATE_20( MPI_Type_get_extent )
int
MPIAPI
MPI_Type_lb(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* displacement
    );
MSMPI_DEPRECATE_20( PMPI_Type_get_extent )
int
MPIAPI
PMPI_Type_lb(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* displacement
    );

MSMPI_DEPRECATE_20( MPI_Type_get_extent )
int
MPIAPI
MPI_Type_ub(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* displacement
    );
MSMPI_DEPRECATE_20( PMPI_Type_get_extent )
int
MPIAPI
PMPI_Type_ub(
    MPI_Datatype datatype,
    _Out_ MPI_Aint* displacement
    );


typedef MPI_Comm_copy_attr_function MPI_Copy_function;
typedef MPI_Comm_delete_attr_function MPI_Delete_function;

#define MPI_NULL_COPY_FN ((MPI_Copy_function*)0)
#define MPI_NULL_DELETE_FN ((MPI_Delete_function*)0)
#define MPI_DUP_FN MPIR_Dup_fn


MSMPI_DEPRECATE_20( MPI_Comm_create_keyval )
int
MPIAPI
MPI_Keyval_create(
    _In_ MPI_Copy_function* copy_fn,
    _In_ MPI_Delete_function* delete_fn,
    _Out_ int* keyval,
    _In_opt_ void* extra_state
    );
MSMPI_DEPRECATE_20( PMPI_Comm_create_keyval )
int
MPIAPI
PMPI_Keyval_create(
    _In_ MPI_Copy_function* copy_fn,
    _In_ MPI_Delete_function* delete_fn,
    _Out_ int* keyval,
    _In_opt_ void* extra_state
    );

MSMPI_DEPRECATE_20( MPI_Comm_free_keyval )
int
MPIAPI
MPI_Keyval_free(
    _Inout_ int* keyval
    );
MSMPI_DEPRECATE_20( PMPI_Comm_free_keyval )
int
MPIAPI
PMPI_Keyval_free(
    _Inout_ int* keyval
    );

MSMPI_DEPRECATE_20( MPI_Comm_set_attr )
int
MPIAPI
MPI_Attr_put(
    MPI_Comm comm,
    int keyval,
    _In_ void* attribute_val
    );
MSMPI_DEPRECATE_20( PMPI_Comm_set_attr )
int
MPIAPI
PMPI_Attr_put(
    MPI_Comm comm,
    int keyval,
    _In_ void* attribute_val
    );

MSMPI_DEPRECATE_20( MPI_Comm_get_attr )
int
MPIAPI
MPI_Attr_get(
    MPI_Comm comm,
    int keyval,
    _Out_ void* attribute_val,
    _Out_ int* flag
    );
MSMPI_DEPRECATE_20( PMPI_Comm_get_attr )
int
MPIAPI
PMPI_Attr_get(
    MPI_Comm comm,
    int keyval,
    _Out_ void* attribute_val,
    _Out_ int* flag
    );

MSMPI_DEPRECATE_20( MPI_Comm_delete_attr )
int
MPIAPI
MPI_Attr_delete(
    MPI_Comm comm,
    int keyval
    );
MSMPI_DEPRECATE_20( PMPI_Comm_delete_attr )
int
MPIAPI
PMPI_Attr_delete(
    MPI_Comm comm,
    int keyval
    );


typedef MPI_Comm_errhandler_fn MPI_Handler_function;

MSMPI_DEPRECATE_20( MPI_Comm_create_errhandler )
int
MPIAPI
MPI_Errhandler_create(
    _In_ MPI_Handler_function* function,
    _Out_ MPI_Errhandler* errhandler
    );
MSMPI_DEPRECATE_20( PMPI_Comm_create_errhandler )
int
MPIAPI
PMPI_Errhandler_create(
    _In_ MPI_Handler_function* function,
    _Out_ MPI_Errhandler* errhandler
    );

MSMPI_DEPRECATE_20( MPI_Comm_set_errhandler )
int
MPIAPI
MPI_Errhandler_set(
    MPI_Comm comm,
    MPI_Errhandler errhandler
    );
MSMPI_DEPRECATE_20( PMPI_Comm_set_errhandler )
int
MPIAPI
PMPI_Errhandler_set(
    MPI_Comm comm,
    MPI_Errhandler errhandler
    );

MSMPI_DEPRECATE_20( MPI_Comm_get_errhandler )
int
MPIAPI
MPI_Errhandler_get(
    MPI_Comm comm,
    _Out_ MPI_Errhandler* errhandler
    );
MSMPI_DEPRECATE_20( PMPI_Comm_get_errhandler )
int
MPIAPI
PMPI_Errhandler_get(
    MPI_Comm comm,
    _Out_ MPI_Errhandler* errhandler
    );


/*---------------------------------------------------------------------------*/
/* Chapter 16: Language Bindings                                             */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------*/
/* Section 16.2: Fortran Support               */
/*---------------------------------------------*/

int
MPIAPI
MPI_Type_create_f90_real(
    int p,
    int r,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_f90_real(
    int p,
    int r,
    _Out_ MPI_Datatype* newtype
    );

int
MPIAPI
MPI_Type_create_f90_complex(
    int p,
    int r,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_f90_complex(
    int p,
    int r,
    _Out_ MPI_Datatype* newtype
    );

int
MPIAPI
MPI_Type_create_f90_integer(
    int r,
    _Out_ MPI_Datatype* newtype
    );
int
MPIAPI
PMPI_Type_create_f90_integer(
    int r,
    _Out_ MPI_Datatype* newtype
    );

/* typeclasses */
#define MPI_TYPECLASS_REAL      1
#define MPI_TYPECLASS_INTEGER   2
#define MPI_TYPECLASS_COMPLEX   3

int
MPIAPI
MPI_Type_match_size(
    int typeclass,
    int size,
    _Out_ MPI_Datatype* type
    );
int
MPIAPI
PMPI_Type_match_size(
    int typeclass,
    int size,
    _Out_ MPI_Datatype* type
    );


/*---------------------------------------------*/
/* Section 16.3: Language Interoperability     */
/*---------------------------------------------*/

#define MPI_Comm_c2f(comm)  (MPI_Fint)(comm)
#define PMPI_Comm_c2f(comm) (MPI_Fint)(comm)

#define MPI_Comm_f2c(comm)  (MPI_Comm)(comm)
#define PMPI_Comm_f2c(comm) (MPI_Comm)(comm)


#define MPI_Type_f2c(datatype)  (MPI_Datatype)(datatype)
#define PMPI_Type_f2c(datatype) (MPI_Datatype)(datatype)

#define MPI_Type_c2f(datatype)  (MPI_Fint)(datatype)
#define PMPI_Type_c2f(datatype) (MPI_Fint)(datatype)


#define MPI_Group_f2c(group)  (MPI_Group)(group)
#define PMPI_Group_f2c(group) (MPI_Group)(group)

#define MPI_Group_c2f(group)  (MPI_Fint)(group)
#define PMPI_Group_c2f(group) (MPI_Fint)(group)


#define MPI_Request_f2c(request)  (MPI_Request)(request)
#define PMPI_Request_f2c(request) (MPI_Request)(request)

#define MPI_Request_c2f(request)  (MPI_Fint)(request)
#define PMPI_Request_c2f(request) (MPI_Fint)(request)


#define MPI_Win_f2c(win)  (MPI_Win)(win)
#define PMPI_Win_f2c(win) (MPI_Win)(win)

#define MPI_Win_c2f(win)  (MPI_Fint)(win)
#define PMPI_Win_c2f(win) (MPI_Fint)(win)


#define MPI_Op_c2f(op)  (MPI_Fint)(op)
#define PMPI_Op_c2f(op) (MPI_Fint)(op)

#define MPI_Op_f2c(op)  (MPI_Op)(op)
#define PMPI_Op_f2c(op) (MPI_Op)(op)


#define MPI_Info_c2f(info)  (MPI_Fint)(info)
#define PMPI_Info_c2f(info) (MPI_Fint)(info)

#define MPI_Info_f2c(info)  (MPI_Info)(info)
#define PMPI_Info_f2c(info) (MPI_Info)(info)


#define MPI_Errhandler_c2f(errhandler)  (MPI_Fint)(errhandler)
#define PMPI_Errhandler_c2f(errhandler) (MPI_Fint)(errhandler)

#define MPI_Errhandler_f2c(errhandler)  (MPI_Errhandler)(errhandler)
#define PMPI_Errhandler_f2c(errhandler) (MPI_Errhandler)(errhandler)


MPI_File
MPIAPI
MPI_File_f2c(
    MPI_Fint file
    );
MPI_File
MPIAPI
PMPI_File_f2c(
    MPI_Fint file
    );

MPI_Fint
MPIAPI
MPI_File_c2f(
    MPI_File file
    );
MPI_Fint
MPIAPI
PMPI_File_c2f(
    MPI_File file
    );

int
MPIAPI
MPI_Status_f2c(
    _In_ MPI_Fint* f_status,
    _Out_ MPI_Status* status
    );
int
MPIAPI
PMPI_Status_f2c(
    _In_ MPI_Fint* f_status,
    _Out_ MPI_Status* status
    );

int
MPIAPI
MPI_Status_c2f(
    _In_ MPI_Status* status,
    _Out_ MPI_Fint* f_status
    );
int
MPIAPI
PMPI_Status_c2f(
    _In_ MPI_Status* status,
    _Out_ MPI_Fint* f_status
    );


#if !defined(_MPICH_DLL_)
#define MPIU_DLL_SPEC __declspec(dllimport)
#else
#define MPIU_DLL_SPEC
#endif

extern MPIU_DLL_SPEC MPI_Fint* MPI_F_STATUS_IGNORE;
extern MPIU_DLL_SPEC MPI_Fint* MPI_F_STATUSES_IGNORE;


/*---------------------------------------------------------------------------*/
/* Implementation Specific                                                   */
/*---------------------------------------------------------------------------*/

int
MPIAPI
MPIR_Dup_fn(
    MPI_Comm oldcomm,
    int keyval,
    _In_opt_ void* extra_state,
    _In_ void* attribute_val_in,
    _Out_ void* attribute_val_out,
    _Out_ int* flag
    );


#if MSMPI_VER >= 0x300

int
MPIAPI
MSMPI_Get_bsend_overhead();

#endif


#if MSMPI_VER >= 0x300

int
MPIAPI
MSMPI_Get_version();

#else
#  define MSMPI_Get_version() (MSMPI_VER)
#endif

typedef void
(MPIAPI MSMPI_Request_callback)(
    _In_ MPI_Status* status
    );

int
MPIAPI
MSMPI_Request_set_apc(
    MPI_Request request,
    _In_ MSMPI_Request_callback* callback_fn,
    _In_ MPI_Status* callback_status
    );

typedef struct _MSMPI_LOCK_QUEUE
{
    struct _MSMPI_LOCK_QUEUE* volatile next;
    volatile MPI_Aint flags;

} MSMPI_Lock_queue;

void
MPIAPI
MSMPI_Queuelock_acquire(
    _Out_ MSMPI_Lock_queue* queue
    );

void
MPIAPI
MSMPI_Queuelock_release(
    _In_ MSMPI_Lock_queue* queue
    );

int
MPIAPI
MSMPI_Waitsome_interruptible(
    int incount,
    _Inout_count_(incount) MPI_Request array_of_requests[],
    _Out_ int* outcount,
    _Out_cap_post_count_(incount,*outcount) int array_of_indices[],
    _Out_cap_post_count_(incount,*outcount) MPI_Status array_of_statuses[]
    );


/*---------------------------------------------------------------------------*/
/* SAL ANNOTATIONS                                                           */
/*---------------------------------------------------------------------------*/

//OACR_WARNING_POP

#ifdef MSMPI_DEFINED_SAL
#undef MSMPI_DEFINED_SAL
#undef _In_
#undef _In_z_
#undef _In_opt_
#undef _In_count_
#undef _In_bytecount_
#undef _In_opt_count_
#undef _Out_
#undef _Out_cap_
#undef _Out_cap_post_count_
#undef _Out_bytecap_
#undef _Out_z_cap_
#undef _Out_z_cap_post_count_
#undef _Out_cap_post_part_
#undef _Out_opt_
#undef _Out_opt_cap_
#undef _Post_z_
#undef _Inout_
#undef _Inout_count_
#endif

#if defined(__cplusplus)
}
#endif

#endif /* MPI_INCLUDED */
