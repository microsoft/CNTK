%module wallkw

// test the -Wallkw option

%warnfilter(SWIGWARN_PARSE_KEYWORD) clone; // 'clone' is a PHP keyword, renaming to 'c_clone'
%warnfilter(SWIGWARN_PARSE_KEYWORD) delegate; // 'delegate' is a C# keyword, renaming to '_delegate'
%warnfilter(SWIGWARN_PARSE_KEYWORD) pass; // 'pass' is a python keyword, renaming to '_pass'
%warnfilter(SWIGWARN_PARSE_KEYWORD) alias; // 'alias' is a D keyword, renaming to '_alias'
%warnfilter(SWIGWARN_PARSE_KEYWORD) rescue; // 'rescue' is a ruby keyword, renaming to 'C_rescue'

%inline %{
const char * clone() { return "clone"; }
const char * delegate() { return "delegate"; }
const char * pass() { return "pass"; }
const char * alias() { return "alias"; }
const char * rescue() { return "rescue"; }
%}

