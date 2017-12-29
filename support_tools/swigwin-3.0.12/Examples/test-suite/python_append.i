/*
Testcase to test %pythonprepend and %pythonappend %pythoncode %pythonbegin
*/

%module python_append 

%pythoncode %{
mypath = os.path.dirname("/a/b/c/d.txt")
funcpath = None
staticfuncpath = None
def grabpath():
    return funcpath
def grabstaticpath():
    return staticfuncpath
def clearstaticpath():
    global staticfuncpath
    staticfuncpath = None
%}

%pythonappend Test::func %{
funcpath = os.path.dirname(funcpath)
%}

%pythonprepend Test::func %{
global funcpath
funcpath = mypath
%}

%pythonappend Test::static_func %{
staticfuncpath = os.path.basename(staticfuncpath)
pass   
%}

%pythonprepend Test::static_func {
global staticfuncpath
staticfuncpath = mypath
pass
}

%pythonbegin %{
import os.path
%}

%inline %{

class Test {
public:
  static void static_func() {};
  void func() {};
};

#ifdef SWIGPYTHON_BUILTIN
bool is_python_builtin() { return true; }
#else
bool is_python_builtin() { return false; }
#endif
%}

