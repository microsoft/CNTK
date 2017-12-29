%module "enum_thorough_typesafe"

// Test enum wrapping using the typesafe enum pattern in the target language
%include "enumtypesafe.swg"

#define SWIG_TEST_NOCSCONST // For C# typesafe enums

%include "enum_thorough.i"

