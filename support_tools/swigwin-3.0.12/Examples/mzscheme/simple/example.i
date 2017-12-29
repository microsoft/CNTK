/* File : example.i */
%module example
%{
/* Put headers and other declarations here */
%}

%include typemaps.i

%rename(mod) my_mod;

%inline %{
extern double My_variable;
extern int    fact(int);
extern int    my_mod(int n, int m);
extern char   *get_time();
%}
