%module python_pythoncode

// github issue#379 - these examples failed with 3.0.5 and earlier (at least as
// far back as 1.3.37):

struct TYPE {
%pythoncode %{
    def one():
        a = 1
# Comment XXXX
        return a
%}
};

%define %bar
%pythoncode %{
    def one():
        a = 1
        #
        return a
%}
%enddef

struct TYPE2 {
%bar
};

%{
struct TYPE { };
struct TYPE2 { };
%}
