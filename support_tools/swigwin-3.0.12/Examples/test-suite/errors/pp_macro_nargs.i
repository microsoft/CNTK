%module xxx

#define foo(a,x)      a x
#define bar(x)        x
#define spam()        /**/

foo(3)
foo(3,4,5)
bar()
bar(2,3)
spam(1)





