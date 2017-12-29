/* -*- c -*- */

%module example;

%{
void divide_l(int a, int b, int *quotient_p, int *remainder_p);
void divide_v(int a, int b, int *quotient_p, int *remainder_p);
void divide_mv(int a, int b, int *quotient_p, int *remainder_p);
%}

/* Multiple values as lists. By default, if more than one value is to
be returned, a list of the values is created and returned; to switch
back to this behavior, use: */
%values_as_list; 

void divide_l(int a, int b, int *OUTPUT, int *OUTPUT);

/* Multiple values as vectors. By issueing: */
%values_as_vector;
/* vectors instead of lists will be used. */

void divide_v(int a, int b, int *OUTPUT, int *OUTPUT);

/* Multiple values for multiple-value continuations.
   (This is the most elegant way.)  By issueing: */
%multiple_values;
/* multiple values are passed to the multiple-value
   continuation, as created by `call-with-values' or the
   convenience macro `receive'. (See the Scheme file.) */

void divide_mv(int a, int b, int *OUTPUT, int *OUTPUT);

