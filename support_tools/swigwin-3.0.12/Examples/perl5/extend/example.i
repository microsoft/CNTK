/* File : example.i */
%module(directors="1") example
%{
#include "example.h"
%}

%include "std_vector.i"
%include "std_string.i"

/* turn on director wrapping for Manager */
%feature("director") Employee;
%feature("director") Manager;

/* EmployeeList::addEmployee(Employee *p) gives ownership of the
 * employee to the EmployeeList object.  The wrapper code should
 * understand this. */
%apply SWIGTYPE *DISOWN { Employee *p };

%include "example.h"

