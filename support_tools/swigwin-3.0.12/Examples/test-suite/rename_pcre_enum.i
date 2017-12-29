%module rename_pcre_enum

// This file is needed for proper enum support in C#/Java backends
#if defined(SWIGCSHARP) || defined(SWIGJAVA)
%include "enums.swg"
#endif

// Apply a rule for renaming the enum elements to avoid the common prefixes
// redundant in C#/Java
%rename("%(regex:/([A-Z][a-z]+)+_(.*)/\\2/)s",%$isenumitem) "";

// Also don't export special end of enum markers which are often used in C++
// code to just have a symbolic name for the number of enum elements but are
// not needed in target language.
%rename("$ignore", regexmatch$name="([A-Z][a-z]+)+_Max$",%$isenumitem) "";

// Test another way of doing the same thing with regextarget:
%rename("$ignore", %$isenumitem, regextarget=1) "([A-Z][a-z]+)+_Internal$";

// Apply this renaming rule to all enum elements that don't contain more than
// one capital letter.
%rename("%(lower)s", notregexmatch$name="[A-Z]\\w*[A-Z]", %$isenumitem) "";

%inline %{

// Foo_Internal and Foo_Max won't be exported.
enum Foo {
    Foo_Internal = -1,
    Foo_First,
    Foo_Second,
    Foo_Max
};

// All elements of this enum will be exported because they do not match the
// excluding regex.
enum BoundaryCondition {
    BoundaryCondition_MinMax,
    BoundaryCondition_MaxMin,
    BoundaryCondition_MaxMax
};

// The elements of this enum will have lower-case names.
enum Colour {
    Red,
    Blue,
    Green
};

%}
