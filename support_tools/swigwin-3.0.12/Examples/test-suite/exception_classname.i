%module exception_classname

%warnfilter(SWIGWARN_RUBY_WRONG_NAME);
#if defined(SWIGPHP) || defined(SWIGD)
%rename(ExceptionClass) Exception;
#endif

%inline %{
class Exception {
public:
  int testfunc() { return 42; }
};
%}
