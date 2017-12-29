%module grouping

%inline %{

typedef int Integer;

int (test1)(int x) {
   return x;
}

int *(test2)(int x) {
    static int y;
    y = x;
    return &y;
}

int (test3) = 37;

typedef Integer (UnaryOp)(Integer);

Integer do_unary(Integer x, UnaryOp *f) {
   return (*f)(x);
}

int negate(int x) {
   return -x;
}

%}

%constant UnaryOp *NEGATE = negate;
