/* File : example.h */

enum error {OK, OVERFLOW, DIVISION_BY_ZERO, NEGATIVE_RADICAND, NEGATIVE_BASE};
typedef error errorstate;  /* just to separate the typemaps */

error acc_add (double &x, double y);
error acc_sub (double &x, double y);
error acc_mul (double &x, double y);
error acc_div (double &x, double y);

double op_add (double x, double y, errorstate &err);
double op_sub (double x, double y, errorstate &err);
double op_mul (double x, double y, errorstate &err);
double op_div (double x, double y, errorstate &err);
double op_sqrt (double x, errorstate &err);
double op_pow (double x, double y, errorstate &err);

double op_noexc (double x, double y);
