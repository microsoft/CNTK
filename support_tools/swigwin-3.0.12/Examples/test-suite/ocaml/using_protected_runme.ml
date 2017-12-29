open Swig
open Using_protected

let f = new_FooBar C_void
let _ = (invoke f) "x" (C_int 3)

let _ = if (invoke f) "blah" (C_int 4) <> (C_int 4) then
    raise (Failure "blah(int)")
