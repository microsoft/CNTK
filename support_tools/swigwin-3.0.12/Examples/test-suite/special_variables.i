%module(directors="1") special_variables

%include <std_string.i>

// will fail to compile if $symname is not expanded
%typemap(argout) int i {
  $symname(99);
}

%{
#define KKK_testmethod testmethod
#define KKK_teststaticmethod KKK::teststaticmethod
%}

%inline %{
void testmethod(int i) {}
struct KKK {
  void testmethod(int i) {}
  static void teststaticmethod(int i) {}
};
%}

%{
std::string ExceptionVars(double i, double j) {
  return "a1";
}
%}

%rename(ExceptionVars) Space::exceptionvars;
%exception Space::exceptionvars %{
  $action
  result = $symname(1.0,2.0); // Should expand to ExceptionVars
  result = $name(3.0,4.0); // Should expand to Space::exceptionvars
  // above will not compile if the variables are not expanded properly
  result = "$action  $name  $symname  $overname $wrapname $parentclassname $parentclasssymname";
%}
%inline %{
namespace Space {
std::string exceptionvars(double i, double j) {
  return "b2";
}
}
%}


%exception Space::overloadedmethod %{
  $action
  result = Space::$symname(1.0);
  result = $name();
  result = $name(2.0);
  // above will not compile if the variables are not expanded properly
  result = "$action  $name  $symname  $overname $wrapname $parentclassname $parentclasssymname";
  // $decl
%}

%inline %{
namespace Space {
  std::string overloadedmethod(double j) {
    return "c3";
  }
  std::string overloadedmethod() {
    return "d4";
  }
}
std::string declaration;
%}

%exception {
  $action
  declaration = "$fulldecl $decl";
}

%inline %{
namespace SpaceNamespace {
  struct ABC {
    ABC(int a, double b) {}
    ABC() {}
    static short * staticmethod(int x, bool b) { return 0; }
    short * instancemethod(int x, bool b = false) { return 0; }
    short * constmethod(int x) const { return 0; }
  };
  template<typename T> struct Template {
    std::string tmethod(T t) { return ""; }
  };
  void globtemplate(Template<ABC> t) {}
}
%}

%template(TemplateABC) SpaceNamespace::Template<SpaceNamespace::ABC>;

/////////////////////////////////// directors /////////////////////////////////
%{
void DirectorTest_director_testmethod(int i) {}
void DirectorTest_director_testmethodSwigExplicitDirectorTest(int i) {}
%}
%typemap(directorargout) int i {
  $symname(99);
}
%feature("director") DirectorTest;
%inline %{
void director_testmethod(int i) {}
struct DirectorTest {
  virtual void director_testmethod(int i) {}
  virtual ~DirectorTest() {}
};
%}


/////////////////////////////////// parentclasssymname parentclassname /////////////////////////////////
%exception instance_def {
  $action
  $parentclasssymname_aaa();
  $parentclassname_bbb();
  // above will not compile if the variables are not expanded properly
}
%exception static_def {
  $action
  $parentclasssymname_aaa();
  $parentclassname_bbb();
  // above will not compile if the variables are not expanded properly
}

%{
void DEFNewName_aaa() {}
namespace SpaceNamespace {
  void DEF_bbb() {}
}
%}

%rename(DEFNewName) DEF;
%inline %{
namespace SpaceNamespace {
  struct DEF : ABC {
    void instance_def() {}
    static void static_def() {}
  };
}
%}

