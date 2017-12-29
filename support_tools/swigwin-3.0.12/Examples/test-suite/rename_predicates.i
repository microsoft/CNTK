%module rename_predicates

// Test a few of the predicates - %$isfunction etc
%rename("AF_%(utitle)s", %$isfunction) "";
%rename("MF_%(utitle)s", %$isfunction, %$ismember) "";
%rename("GF_%(utitle)s", %$isfunction, %$not %$ismember) "";
%rename("MV_%(utitle)s", %$isvariable) "";
%rename("GV_%(utitle)s", %$isvariable, %$isglobal) "";

%extend RenamePredicates {
  void extend_function_before() {}
}

%inline %{
struct RenamePredicates {
  RenamePredicates(int v = 0) : member_variable(v) {}
  void member_function() {}
  static void static_member_function() {}
  int member_variable;
  static int static_member_variable;
};
int RenamePredicates::static_member_variable = 456;
int global_variable = 789;
void global_function() {}
%}

%extend RenamePredicates {
  void extend_function_after() {}
}

// Test the various %rename functions - %(upper) etc
%rename("UC_%(upper)s") "uppercase";
%rename("LC_%(lower)s") "LOWERcase";
%rename("TI_%(title)s") "title";
%rename("FU_%(firstuppercase)s") "firstUpperCase";
%rename("FL_%(firstlowercase)s") "FirstLowerCase";
%rename("CA_%(camelcase)s") "camel_Case";
%rename("LC_%(lowercamelcase)s") "Lower_camel_Case";
%rename("UC_%(undercase)s") "UnderCaseIt";

%inline %{
void uppercase() {}
void LOWERcase() {}
void title() {}
void firstUpperCase() {}
void FirstLowerCase() {}
void camel_Case() {}
void Lower_camel_Case() {}
void UnderCaseIt() {}
%}

// Test renaming only member functions in %extend
%rename("EX_%(upper)s", %$isfunction, %$isextendmember) "";
%extend ExtendCheck {
  void ExtendMethod1() {}
}
%inline %{
struct ExtendCheck {
  void RealMember1() {}
#ifdef SWIG
  %extend {
    void ExtendMethod2() {}
  }
#endif
  void RealMember2() {}
};
%}
%extend ExtendCheck {
  void ExtendMethod3() {}
}

