/* File : example.i */
%module python_destructor_exception
%include exception.i

%exception ClassWithThrowingDestructor::~ClassWithThrowingDestructor()
{
  $action
  SWIG_exception(SWIG_RuntimeError, "I am the ClassWithThrowingDestructor dtor doing bad things");
}

%inline %{
class ClassWithThrowingDestructor
{
};

%}

%include <std_vector.i>
%template(VectorInt) std::vector<int>;
