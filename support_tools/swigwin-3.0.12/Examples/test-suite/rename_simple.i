%module rename_simple

%rename(NewStruct)           OldStruct;
%rename(NewVariable)         OldVariable;
%rename(NewInstanceMethod)   OldInstanceMethod;
%rename(NewInstanceVariable) OldInstanceVariable;
%rename(NewStaticMethod)     OldStaticMethod;
%rename(NewStaticVariable)   OldStaticVariable;
%rename(NewFunction)         OldFunction;
%rename(NewGlobalVariable)   OldGlobalVariable;

%inline %{
struct OldStruct {
  enum { ONE = 1, TWO, THREE };
  OldStruct() : OldInstanceVariable(111) {}
  int OldInstanceVariable;
  int OldInstanceMethod() { return 222; }
  static int OldStaticVariable;
  static int OldStaticMethod() { return 333; }
};
int OldStruct::OldStaticVariable = 444;

int OldFunction() { return 555; }
int OldGlobalVariable = 666;
%}
