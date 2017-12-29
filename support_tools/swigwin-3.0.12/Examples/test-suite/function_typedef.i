%module function_typedef

%inline %{

typedef int binop_t(int, int);

int do_binop1(binop_t f, int x, int y) {
   return f(x,y);
}

int do_binop2(binop_t *f, int x, int y) {
   return (*f)(x,y);
}

int do_binop3(int f(int,int), int x, int y) {
   return f(x,y);
}
%}


