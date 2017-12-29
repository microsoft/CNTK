%module template_opaque
%include "std_vector.i"

%{
  namespace A 
  {
    struct OpaqueStruct 
    {
      int x;
    };
  }

  enum Hello { hi, hello };
      
%}


%inline {
namespace A {
  struct OpaqueStruct;
  typedef struct OpaqueStruct OpaqueType;
  typedef enum Hello Hi;
  typedef std::vector<OpaqueType> OpaqueVectorType;
  typedef std::vector<Hi> OpaqueVectorEnum;
  
  void FillVector(OpaqueVectorType& v) 
  {
    for (size_t i = 0; i < v.size(); ++i) {
      v[i] = OpaqueStruct();
    }
  }

  void FillVector(const OpaqueVectorEnum& v) 
  {
  }
}
}

#ifndef SWIGCSHARP
// C# vector typemaps only ready for simple cases right now
%template(OpaqueVectorType) std::vector<A::OpaqueType>; 
#endif
