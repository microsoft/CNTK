%module template_extend_overload


%inline %{
  template <class T>
    struct A
    {
    };

  template <class Arg, class Res>
    struct B
    {
    };
%}


%define __compose_unary(Class, ArgType, ResType)
  Class<ResType> compose(const B<ArgType, ResType>& f)
  {
    return Class<ResType>();
  }
%enddef

%define __compose_unary_3(Class, Type)
%extend Class<Type>
{
  __compose_unary(Class, Type, bool);
  __compose_unary(Class, Type, double);
  __compose_unary(Class, Type, int);
}
%enddef

%define compose_unary(Class)
  __compose_unary_3(Class, bool)
  __compose_unary_3(Class, double)
  __compose_unary_3(Class, int)
%enddef

compose_unary(A);
  
%template(A_double) A<double>;
%template(A_int) A<int>;
%template(A_bool) A<bool>;

