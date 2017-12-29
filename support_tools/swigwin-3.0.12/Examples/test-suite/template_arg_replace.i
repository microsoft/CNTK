%module template_arg_replace

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) test_Matrix<float, 3, 3>;	/* Ruby, wrong class name */

%inline %{

template <typename T, int r, int c> class test_Matrix { 
public: 
 void Func(const test_Matrix<T,r,c> &m) { }; 
}; 
%} 

%template (matrix33f) test_Matrix<float,3, 3>; 

