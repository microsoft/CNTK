%module sneaky1

%{
int add(int x, int y) {
   return x+y;
}

int subtract(int x, int y) {
   return x-y;
}
int mul(int x, int y) {
   return x*y;
}

int divide(int x, int y) {
   return x/y;
}
%}

%inline %{
typedef int binop(int,int);
%}

binop add,subtract,mul,divide;





