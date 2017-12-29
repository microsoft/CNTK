/* File : example.i */
%module Example
%{
/* Put headers and other declarations here */
%}

%inline %{
extern double My_variable;
extern int    fact(int);
extern int    mod(int n, int m);
extern char  *get_time();
%}

%include guile/guilemain.i
