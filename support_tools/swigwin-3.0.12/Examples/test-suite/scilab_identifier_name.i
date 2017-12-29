%module scilab_identifier_name

//
// Test long identifier name (> 24 characters) truncating
//

// Test truncating variables, constants, functions identifier names

%inline %{
// these identifier names wont be truncated
int gvar_identifier_name = -1;
#define CONS_IDENTIFIER_NAME -11
int function_identifier_name() { return -21; };

// these identifier names will be truncated
int too_long_gvar_identifier_name_1 = 1;
int too_long_gvar_identifier_name_2 = 2;

#define TOO_LONG_CONST_IDENTIFIER_NAME_1 11

int too_long_function_identifier_name_1() { return 21; };
%}

// Test truncating when %scilabconst mode is activated
%scilabconst(1);

%inline %{
#define SC_CONST_IDENTIFIER_NAME -12;

#define SC_TOO_LONG_CONST_IDENTIFIER_NAME_1 13
#define SC_TOO_LONG_CONST_IDENTIFIER_NAME_2 14
%}
%scilabconst(0);

// Test truncating in the case of struct
%inline %{
struct st {
  int m_identifier_name;
  int too_long_member_identifier_name;
};

%}











