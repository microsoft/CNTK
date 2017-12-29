%module conversion_operators

// Test bug #401 where the conversion operator name incorrectly included the newline character
// Also test comments around conversion operators due to special handling in the scanner for conversion operators

// These one line ignores should match the conversion operator names to suppress Warning 503 - SWIGWARN_LANG_IDENTIFIER
%ignore operator const EcReal;
%ignore operator EcImaginary const;
%ignore operator EcComplex const;

%inline %{

struct EcReal {};
struct EcImaginary {};
struct EcComplex {};

struct EcAngle {
   operator const EcReal
      (
      ) const;
   operator EcImaginary
const (
      ) const;
   operator
EcComplex
      const (
      ) const;
};

struct EcAngle2 {
   operator const EcReal/* C comment */
      (
      ) const;
   operator EcImaginary/* C comment */
const (
      ) const;
   operator/* C comment */
EcComplex
      const (
      ) const;
};

struct EcAngle3 {
   operator const EcReal // C++ comment
      (
      ) const;
   operator EcImaginary // C++ comment
const (
      ) const;
   operator // C++ comment
EcComplex
      const (
      ) const;
};
%}
