%module namespace_union

#pragma SWIG nowarn=SWIGWARN_PARSE_UNNAMED_NESTED_CLASS

%inline %{
namespace SpatialIndex 
{ 
        class Variant 
        { 
        public: 
                Variant() { }; 
                int varType; 
                union 
                { 
                        long           lVal;         // VT_LONG 
                        short          iVal;         // VT_SHORT 
                        float          fltVal;       // VT_FLOAT 
                        double         dblVal;       // VT_DOUBLE 
                        char           cVal;         // VT_CHAR 
                        unsigned short uiVal;        // VT_USHORT 
                        unsigned long  ulVal;        // VT_ULONG 
                        int            intVal;       // VT_INT 
                        unsigned int   uintVal;      // VT_UINT 
                        bool           blVal;        // VT_BOOL 
                        char*          pcVal;        // VT_PCHAR 
                        void*          pvVal;        // VT_PVOID 
                } val; 
        }; // Variant 
} 
%}
 
