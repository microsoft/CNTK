/* This file is part of a test for SF bug #231619. 
   It shows that the %import directive does not work properly in SWIG
   1.3a5:  Type information is not properly generated if a base class
   comes from an %import-ed file. 

   Extra tests added for enums to test languages that have enum types.
*/

%module imports_b

%{ 
#include "imports_b.h" 
%} 


/* 
   To import, you can use either
   
     %import "imports_a.i"

   or

     %import(module="imports_a") "imports_a.h" 


   In the first case, imports_a.i should declare the module name using
   the %module directive.

   In the second case, the file could be either a .h file, where no
   %module directive will be found, or a swig interface file, where
   the module option will take priority over any %module directive
   inside the imported file.

*/

#if 0
  %import "imports_a.i"
#else
#  if 0
  // Test Warning 401 (Python only)
  %import "imports_a.h" 
#  else
  %import(module="imports_a") "imports_a.h" 
#  endif
#endif

%include "imports_b.h"  
