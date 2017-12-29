%module naturalvar_onoff

// Test naturalvar feature override is working -
// naturalvar on the variable name has priority over naturalvar on the variable's type
// Use runtime tests to differentiate between the const ref typemaps and pointer typemap -
// using the fact that NULL cannot be passed to the ref typemaps

%naturalvar Member1;
%nonaturalvar Member2;
%naturalvar Member3;
%nonaturalvar Vars::member3Off;
%nonaturalvar Member4;
%naturalvar Vars::member4On;

%inline %{
struct Member1 {};
struct Member2 {};
struct Member3 {};
struct Member4 {};

struct Vars {
    Member1 member1On;
    Member2 member2Off;
    Member3 member3Off;
    Member3 member3On;
    Member4 member4Off;
    Member4 member4On;
};
%}
