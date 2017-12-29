%module global_ns_arg

%inline %{

typedef int Integer;

::Integer foo(::Integer x) {
   return x;
}

::Integer bar_fn() {
   return 1;
}

%}
