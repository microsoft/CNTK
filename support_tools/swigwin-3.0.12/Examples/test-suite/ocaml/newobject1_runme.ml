open Swig
open Newobject1

exception RuntimeError of string * int

let foo1 = ref (_Foo_makeFoo C_void)
let _ = if get_int (_Foo_fooCount C_void) != 1 then
   raise (RuntimeError ("(1) Foo.fooCount != 1",
			get_int (_Foo_fooCount C_void)))

let foo2 = ref ((invoke !foo1) "makeMore" C_void) 
let _ = if get_int (_Foo_fooCount C_void) != 2 then
   raise (RuntimeError ("(2) Foo.fooCount != 2",
			get_int (_Foo_fooCount C_void)))

let _ = begin
  foo1 := C_void ; Gc.full_major () ;
  (if get_int (_Foo_fooCount C_void) != 1 then
	raise (RuntimeError ("(3) Foo.fooCount != 1",
			     get_int (_Foo_fooCount C_void)))) ;

  foo2 := C_void ; Gc.full_major () ;
  (if get_int (_Foo_fooCount C_void) != 0 then
	raise (RuntimeError ("(4) Foo.fooCount != 0",
			     get_int (_Foo_fooCount C_void)))) ;
end
