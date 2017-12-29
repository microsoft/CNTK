var cpp_namespace = require("cpp_namespace");

var n = cpp_namespace.fact(4);
if (n != 24){
    throw ("Bad return value error!");
}
if (cpp_namespace.Foo != 42){
    throw ("Bad variable value error!");
}

t = new cpp_namespace.Test();
if (t.method() != "Test::method"){
    throw ("Bad method return value error!");
}
if (cpp_namespace.do_method(t) != "Test::method"){
    throw ("Bad return value error!");
}

if (cpp_namespace.do_method2(t) != "Test::method"){
    throw ("Bad return value error!");
}
cpp_namespace.weird("hello", 4);
delete t;

t2 = new cpp_namespace.Test2();
t3 = new cpp_namespace.Test3();
t4 = new cpp_namespace.Test4();
t5 = new cpp_namespace.Test5();
if (cpp_namespace.foo3(42) != 42){
    throw ("Bad return value error!");
}

if (cpp_namespace.do_method3(t2,40) != "Test2::method"){
    throw ("Bad return value error!");
}

if (cpp_namespace.do_method3(t3,40) != "Test3::method"){
    throw ("Bad return value error!");
}

if (cpp_namespace.do_method3(t4,40) != "Test4::method"){
    throw ("Bad return value error!");
}

if (cpp_namespace.do_method3(t5,40) != "Test5::method"){
    throw ("Bad return value error!");
}
