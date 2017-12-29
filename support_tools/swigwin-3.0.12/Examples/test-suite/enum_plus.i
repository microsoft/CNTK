%module enum_plus

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) iFoo; /* Ruby, wrong constant name */

%inline %{
struct iFoo 
{ 
    enum { 
       Phoo = +50 
    }; 
}; 
%}
