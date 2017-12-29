/* This testcase checks whether SWIG correctly parses the 'thread_local' storage specifier */

%module cpp11_thread_local

%inline %{
struct ThreadLocals {
//  thread_local int tval; // members must also be declared static
  static thread_local int stval;
  thread_local static int tsval;
  static thread_local const int stcval88 = 88;
  thread_local static const int tscval99 = 99;
};
thread_local int tval;
static thread_local int stval;
thread_local static int tsval;
extern thread_local int etval;
thread_local extern int teval;
extern "C" thread_local int ectval;
extern "C++" thread_local int ecpptval;

thread_local int ThreadLocals::stval = 11;
thread_local int ThreadLocals::tsval = 22;
thread_local const int ThreadLocals::stcval88;
thread_local const int ThreadLocals::tscval99;
%}

%{
// externs
thread_local int etval = 33;
thread_local int teval = 44;
thread_local int ectval = 55;
thread_local int ecpptval = 66;
%}
