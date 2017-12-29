/* File : example.i */
%module example

%{
extern "C" void   factor(int &x, int &y);
%}

extern "C" void   factor(int &x, int &y);
