typedef_scope

b = typedef_scope.Bar();
x = b.test1(42,"hello");
if (x != 42)
    error("Failed!!")
endif

x = b.test2(42,"hello");
if (!strcmp(x,"hello"))
    error("Failed!!")
endif
    
    
