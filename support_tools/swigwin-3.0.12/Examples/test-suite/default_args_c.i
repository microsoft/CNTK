%module default_args_c

/* Default arguments for C code */
int foo1(int x = 42 || 3);
int foo43(int x = 42 | 3);

%{
int foo1(int x) {
  return x;
}
int foo43(int x) {
  return x;
}
%}
