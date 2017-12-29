%module li_std_combinations

%include <std_vector.i>
%include <std_string.i>
%include <std_pair.i>

%template(VectorInt) std::vector<int>;
%template(VectorString) std::vector<std::string>;
%template(PairIntString) std::pair<int, std::string>;

%template(VectorPairIntString) std::vector< std::pair<int, std::string> >;
%template(VectorVectorString) std::vector< std::vector<std::string> >;

#if !defined(SWIGSCILAB)
%template(PairIntVectorString) std::pair< int, std::vector<std::string> >;
%template(PairIntPairIntString) std::pair< int, std::pair<int, std::string> >;
#else
%template(PairIntVecStr) std::pair< int, std::vector<std::string> >;
%template(PairIntPairIntStr) std::pair< int, std::pair<int, std::string> >;
#endif


#if defined(SWIGCSHARP) || defined(SWIGD)
// Checks macro containing a type with a comma
SWIG_STD_VECTOR_ENHANCED(std::pair< double, std::string >)
#endif

%template(PairDoubleString) std::pair< double, std::string >;
%template(VectorPairDoubleString) std::vector< std::pair<double, std::string> >;


