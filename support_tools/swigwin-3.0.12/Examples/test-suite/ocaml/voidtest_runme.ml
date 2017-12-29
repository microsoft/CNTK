open Swig
open Voidtest

let _ = _globalfunc C_void 
let f = new_Foo C_void
let _ = (invoke f) "memberfunc" C_void

let _ = _Foo_staticmemberfunc C_void
