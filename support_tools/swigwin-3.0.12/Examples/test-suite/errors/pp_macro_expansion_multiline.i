%module xxx
// Testing macros split over multiple lines - ensure the warning message for the ignored functions contain the correct line numbering

#define MYMACRO(NAME, A, B, C) void NAME(int A, int B, int C);

MYMACRO(funk, x,
y,

z
)

void foo(int *);
void foo(const int *);

%define MYSWIGMACRO(A, B, C)
MYMACRO(funk1, 
        AA,
        BB,
        CC)
MYMACRO(funk2, 
        AA, 
        BB, 
        CC)
%enddef

MYSWIGMACRO(xx,
 yy,
 zz)

void bar(int *);
void bar(const int *);

